from pathlib import Path

import torch
import numpy as np
import random
import pickle
from absl import logging
from absl.flags import FLAGS
from cellot_model import losses
from cellot_model.utils.loaders import load
from cellot_model.models.cellot import compute_loss_f, compute_loss_g, compute_w2_distance
from cellot_model.train.summary import Logger
from cellot_model.data.utils import cast_loader_to_iterator
from cellot_model.models.ae import compute_scgen_shift
from cellot_model.models import load_autoencoder_model, load_cellfm_model
from cellot_model.utils import load_config
from tqdm import trange
import mindspore as ms


def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None

    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path)
    if key not in ckpt:
        logging.warn(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]


def train_cellot(outdir, config):
    def state_dict(f, g, opts, **kwargs):
        state = {
            "g_state": g.state_dict(),
            "f_state": f.state_dict(),
            "opt_g_state": opts.g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
        }
        state.update(kwargs)

        return state

    def evaluate():
        target = next(iterator_test_target)
        source = next(iterator_test_source)
        source = source.to(device)
        source.requires_grad_(True)
        transport = g.transport(source)

        transport = transport.detach()
        with torch.no_grad():

            if "ae_emb" in config.data:
                target = target.to(device)
                recon = ae_model.decode(transport).detach().cpu().numpy()
                recon_target = ae_model.decode(target).detach().cpu().numpy()
                target = target.detach().cpu()
                transport = transport.cpu().numpy()
            elif "cellfm_emb" in config.data:
                transport = transport.cpu().numpy()
                transport2 = ms.Tensor(transport)
                target2 = ms.Tensor(target.numpy())
                recon = cellfm_model.cellwise_dec(transport2).asnumpy()
                recon_target = cellfm_model.cellwise_dec(target2).asnumpy()
            # else:
            #     recon = transport.cpu().numpy()
            #     treated = target.numpy().mean(0)

            pcc = np.corrcoef(recon.mean(0), treated.mean(0))[0, 1]
            mse = np.linalg.norm(treated.mean(0) - recon.mean(0))



            # pcc_mar = 0
            # mse_mar = 0
            pcc_mar = np.corrcoef(recon.mean(0)[indices], treated.mean(0)[indices])[0, 1]
            mse_mar = np.linalg.norm(treated.mean(0)[[indices]] - recon.mean(0)[[indices]])
            # pcc_mar = np.corrcoef(transport.mean(0), target.numpy().mean(0))[0, 1]
            # pcc_mar = np.linalg.norm(recon_target.mean(0) - treated.mean(0))
            # mse_mar = np.linalg.norm(transport.mean(0) - target.numpy().mean(0))

        # log to logger object
        logger.log(
            "eval",
            pcc=pcc,
            mse=mse,
            step=step,
        )

        return pcc, mse, pcc_mar, mse_mar

    logger = Logger(outdir / "cache/")
    cachedir = outdir / "cache"
    (f, g), opts, loader = load(config)
    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_iters = config.training.n_iters
    # step = load_item_from_save(cachedir / "last.pt", "step", 0)
    step = 0

    # minmmd = load_item_from_save(cachedir / "model.pt", "minmmd", np.inf)
    # mmd = minmmd
    minmse = 10000
    mse = minmse
    import anndata
    adata = anndata.read(config.data.path)
    adata.X = adata.X.todense()
    treated = adata[(adata.obs[config.data.condition] == config.data.target) & (adata.obs['split']=='test')].X

    b = list(adata.var.index)
    key = f'marker_genes-{config.data.condition}-rank'
    a = adata.varm[key][config.data.target].sort_values()
    a = list(a.index[:50])
    indices = []
    for item in a:
        indices.append(b.index(item))

    if "ae_emb" in config.data:
        path_ae = Path(config.data.ae_emb.path)
        model_kwargs = {"input_dim": 798}
        config_ae = load_config(path_ae / "config.yaml")
        ae_model, _ = load_autoencoder_model(
            config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
        )
        ae_model = ae_model.to(device)
    elif "cellfm_emb" in config.data:
        cellfm_model, geneset = load_cellfm_model(config.data.target)


    if 'pair_batch_on' in config.training:
        keys = list(iterator.train.target.keys())
        test_keys = list(iterator.test.target.keys())
    else:
        keys = None

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        if 'pair_batch_on' in config.training:
            assert keys is not None
            key = random.choice(keys)
            iterator_train_target = iterator.train.target[key]
            iterator_train_source = iterator.train.source[key]
            try:
                iterator_test_target = iterator.test.target[key]
                iterator_test_source = iterator.test.source[key]
            # in the iid mode of the ood setting,
            # train and test keys are not necessarily the same ...
            except KeyError:
                test_key = random.choice(test_keys)
                iterator_test_target = iterator.test.target[test_key]
                iterator_test_source = iterator.test.source[test_key]

        else:
            iterator_train_target = iterator.train.target
            iterator_train_source = iterator.train.source
            iterator_test_target = iterator.test.target
            iterator_test_source = iterator.test.source

        target = next(iterator_train_target)
        target = target.to(device)
        for _ in range(config.training.n_inner_iters):
            source = next(iterator_train_source).requires_grad_(True)
            source = source.to(device)

            opts.g.zero_grad()
            gl = compute_loss_g(f, g, source).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()

            gl.backward()
            opts.g.step()

        source = next(iterator_train_source).requires_grad_(True)
        source = source.to(device)

        opts.f.zero_grad()
        fl = compute_loss_f(f, g, source, target).mean()
        fl.backward()
        opts.f.step()
        check_loss(gl, fl)
        f.clamp_w()

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", gloss=gl.item(), floss=fl.item(), step=step)

        if step % config.training.eval_freq == 0:
            pcc, mse, pcc_mar, mse_mar = evaluate()
            print('step: %d PCC: %.4f, mse: %.4f PCC_mar: %.4f, mse_mar: %.4f' % (step, pcc, mse, pcc_mar, mse_mar))
            if mse < minmse:
                minmse = mse
                torch.save(
                    state_dict(f, g, opts, step=step, minmse=minmse),
                    cachedir / "model.pt",
                )

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

            logger.flush()

    torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

    logger.flush()

    return


def train_auto_encoder(outdir, config):
    def state_dict(model, optim, **kwargs):
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        state.update(kwargs)

        return state

    def evaluate(vinputs):
        with torch.no_grad():
            vinputs = vinputs.to(device)
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)

            recon, code = model.outputs(vinputs)
            recon = recon.detach().cpu().numpy()
            label = vinputs.detach().cpu().numpy()
            # print('l2', np.linalg.norm(recon.mean(0) - label.mean(0)))

        return loss

    logger = Logger(outdir / "cache/")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training.n_iters
    # step = load_item_from_save(cachedir / "last.pt", "step", 0)
    step = 0
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    # best_eval_loss = load_item_from_save(
    #     cachedir / "model.pt", "best_eval_loss", np.inf
    # )
    best_eval_loss = np.inf

    import anndata
    adata = anndata.read(config.data.path)
    adata.X = adata.X.todense()
    treated = adata[(adata.obs[config.data.condition] == config.data.target) & (adata.obs['split']=='test')].X
    treated = torch.tensor(treated)
    eval_loss = best_eval_loss

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:

        model.train()
        inputs = next(iterator.train)
        inputs = inputs.to(device)
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", loss=loss.item(), step=step, **comps)

        if step % config.training.eval_freq == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test))
            # eval_loss = evaluate(treated)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)

                torch.save(sd, cachedir / "model.pt")

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")

            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model.name == "scgen" and config.get("compute_scgen_shift", True):
        labels = loader.train.dataset.adata.obs[config.data.condition]
        compute_scgen_shift(model, loader.train.dataset, labels=labels, device=device)

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")

    logger.flush()


def train_popalign(outdir, config):
    def evaluate(config, data, model):

        # Get control and treated subset of the data and projections.
        idx_control_test = np.where(data.obs[
            config.data.condition] == config.data.source)[0]
        idx_treated_test = np.where(data.obs[
            config.data.condition] == config.data.target)[0]

        predicted = transport_popalign(model, data[idx_control_test].X)
        target = np.array(data[idx_treated_test].X)

        # Compute performance metrics.
        mmd = losses.compute_scalar_mmd(target, predicted)
        wst = losses.wasserstein_loss(target, predicted)

        # Log to logger object.
        logger.log(
            "eval",
            mmd=mmd,
            wst=wst,
            step=1
        )

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"

    # Load dataset and previous model parameters.
    model, _, dataset = load(config, restore=cachedir / "last.pt",
                             return_as="dataset")
    train_data = dataset["train"].adata
    test_data = dataset["test"].adata

    if not all(k in model for k in ("dim_red", "gmm_control", "response")):

        if config.model.embedding == 'onmf':
            # Find best low dimensional representation.
            q, nfeats, errors = onmf(train_data.X.T)
            W, proj = choose_featureset(
                train_data.X.T, errors, q, nfeats, alpha=3, multiplier=3)

        else:
            W = np.eye(train_data.X.shape[1])
            proj = train_data.X

        # Get control and treated subset of the data and projections.
        idx_control_train = np.where(train_data.obs[
            config.data.condition] == config.data.source)[0]
        idx_treated_train = np.where(train_data.obs[
            config.data.condition] == config.data.target)[0]

        # Compute probabilistic model for control and treated population.
        gmm_control = build_gmm(
            train_data.X[idx_control_train, :].T,
            proj[idx_control_train], ks=(3), niters=2,
            training=.8, criteria='aic')
        gmm_treated = build_gmm(
            train_data.X[idx_treated_train, :].T,
            proj[idx_treated_train], ks=(3), niters=2,
            training=.8, criteria='aic')

        # Compute alignment between components of both mixture models.
        align, _ = align_components(gmm_control, gmm_treated, method="ref2test")

        # Compute perturbation response for each control component.
        res = get_perturbation_response(align, gmm_control, gmm_treated)

        # Save all results to state dict.
        model = {"dim_red": W,
                 "gmm_control": gmm_control,
                 "gmm_treated": gmm_treated,
                 "response": res}
        state_dict = model
        pickle.dump(state_dict, open(cachedir / "last.pt", 'wb'))
        pickle.dump(state_dict, open(cachedir / "model.pt", 'wb'))

    else:
        W = model["dim_red"]
        gmm_control = model["gmm_control"]
        gmm_treated = model["gmm_treated"]
        res = model["response"]

    # Evaluate performance on test set.
    evaluate(config, test_data, model)
