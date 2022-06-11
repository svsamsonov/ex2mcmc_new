import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from tqdm import tqdm

from . import classification as cl
from . import mh
from .mhgan_utils import discriminator_analysis


def validate_scores(scores):
    assert isinstance(scores, dict)
    for sv in scores.values():
        assert isinstance(sv, np.ndarray)
        assert sv.dtype.kind == "f"
        assert sv.ndim == 1
        # assert np.all(0 <= sv) and np.all(sv <= 1)
    scores = pd.DataFrame(scores)
    return scores


def validate_X(X):
    assert isinstance(X, np.ndarray)
    assert X.dtype.kind == "f"
    batch_size, dim = X.shape
    assert X.shape == (batch_size, dim)
    assert np.all(np.isfinite(X))
    return X


def validate(R):
    """
    X : ndarray, shape (batch_size, nc, image_size, image_size)
    scores : dict of str -> ndarray of shape (batch_size,)
    """
    X, scores = R
    X = validate_X(X)
    scores = validate_scores(scores)
    assert len(X) == len(scores)
    return X, scores


def batched_gen_and_disc(gen_and_disc, n_batches, batch_size):
    """
    Get a large batch of images. Pytorch might run out of memory if we set
    the batch size to n_images=n_batches*batch_size directly.
    g_d_f : callable returning (X, scores) compliant with `validate`
    n_images : int
        assumed to be multiple of batch size
    """
    X, scores = zip(
        *(validate(gen_and_disc(batch_size)) for _ in range(n_batches)),
    )
    X = np.concatenate(X, axis=0)
    scores = pd.concat(scores, axis=0, ignore_index=True)
    return X, scores


def enhance_samples(scores_df, scores_real_df, clf_df, pickers):
    """
    Return selected image (among a batch on n images) for each picker.
    scores_df : DataFrame, shape (n, n_discriminators)
    scores_real_df : DataFrame, shape (m, n_discriminators)
    clf_df : Series, shape (n_classifiers x n_calibrators,)
    pickers : dict of str -> callable
    """
    assert len(scores_df.columns.names) == 1
    assert list(scores_df.columns) == list(scores_real_df.columns)

    init_idx = np.random.choice(len(scores_real_df))

    picked = pd.DataFrame(
        data=0,
        index=pickers.keys(),
        columns=clf_df.index,
        dtype=int,
    )
    cap_out = pd.DataFrame(
        data=False,
        index=pickers.keys(),
        columns=clf_df.index,
        dtype=bool,
    )
    alpha = pd.DataFrame(
        data=np.nan,
        index=pickers.keys(),
        columns=clf_df.index,
        dtype=float,
    )
    for disc_name in sorted(scores_df.columns):
        assert isinstance(disc_name, str)
        s0 = scores_real_df[disc_name].values[init_idx]
        assert np.ndim(s0) == 0
        for calib_name in sorted(clf_df[disc_name].index):
            assert isinstance(calib_name, str)
            # print(f"calibrator name = {calib_name}, discriminator name = {disc_name}")
            calibrator = clf_df[(disc_name, calib_name)]
            s_ = np.concatenate(([s0], scores_df[disc_name].values))
            s_ = calibrator.predict(s_)
            # s_max, = calibrator.predict(np.array([scores_max[disc_name]]))
            for picker_name in sorted(pickers.keys()):
                assert isinstance(picker_name, str)
                # print(f"picker name = {picker_name}")
                # idx, aa = pickers[picker_name](s_, score_max=s_max)
                idx, aa = pickers[picker_name](s_)

                if idx == 0:
                    # Try again but init from first fake
                    cap_out.loc[picker_name, (disc_name, calib_name)] = True
                    # idx, aa = pickers[picker_name](s_[1:], score_max=s_max)
                    idx, aa = pickers[picker_name](s_[1:])
                else:
                    idx = idx - 1
                assert idx >= 0

                picked.loc[picker_name, (disc_name, calib_name)] = idx
                alpha.loc[picker_name, (disc_name, calib_name)] = aa
    return picked, cap_out, alpha


def enhance_samples_series(
    g_d_f,
    scores_real_df,
    clf_df,
    pickers,
    n_samples=16,
    batch_size=64,
    chain_batches=10,
):
    """
    Call enhance_samples multiple times to build up a batch of selected images.
    Stores list of used images X separate from the indices of the images
    selected by each method. This is more memory efficient if there are
    duplicate images selected.
    g_d_f : callable returning (X, scores) compliant with `validate`
    calibrator : dict of str -> trained sklearn classifier
        same keys as scores
    n_images : int
    """
    # batch_size = 16   # Batch size to use when calling the pytorch generator G
    # chain_batches = 10  # Number of batches to use total for the pickers
    # max_est_batches = 156  # Num batches for estimating M in DRS pilot samples

    assert n_samples > 0

    # scores_max = np.inf

    # print('max scores')
    # print(scores_max.to_string())

    X = []
    picked = [None] * n_samples
    cap_out = [None] * n_samples
    alpha = [None] * n_samples
    picked_ = None
    picked_num = 0
    all_generated_num = 0
    for nn in tqdm(range(n_samples)):
        X_, scores_fake_df = batched_gen_and_disc(
            g_d_f,
            chain_batches,
            batch_size,
        )
        # print(f"Shape of generated random images = {X_.shape}")
        # picked_, cc, aa = \
        #     enhance_samples(scores_fake_df, scores_max,
        #                    scores_real_df, clf_df,
        #                     pickers=pickers)
        # print(f"Shape of generated random images = {X_.shape}")
        picked_, cc, aa = enhance_samples(
            scores_fake_df,
            scores_real_df,
            clf_df,
            pickers=pickers,
        )
        picked_ = picked_.unstack()  # Convert to series

        # Only save the used images for memory, so some index x-from needed
        assert np.ndim(picked_.values) == 1
        used_idx, idx_new = np.unique(picked_.values, return_inverse=True)
        picked_ = pd.Series(data=idx_new, index=picked_.index)

        # A bit of index manipulation in our memory saving scheme
        picked[nn] = len(X) + picked_
        add_X = list(X_[used_idx])
        picked_num += len(add_X)
        all_generated_num += len(X_)
        # print(f"number of selected images = {len(add_X)} out of {len(X_)}")

        X.extend(add_X)  # Unravel first index to list
        cap_out[nn] = cc.unstack()
        alpha[nn] = aa.unstack()

    # acceptence_rate = picked_num/all_generated_num
    # print(f"acceptance rate = {acceptence_rate}")
    X = np.asarray(X)
    # assert X.ndim == 4
    picked = pd.concat(picked, axis=1).T
    assert picked.shape == (n_samples, len(picked_))
    cap_out = pd.concat(cap_out, axis=1).T
    assert cap_out.shape == (n_samples, len(picked_))
    alpha = pd.concat(alpha, axis=1).T
    assert alpha.shape == (n_samples, len(picked_))
    return X, picked, cap_out, alpha


@torch.no_grad()
def mh_sampling(
    X_train,
    G,
    D,
    device,
    n_calib_pts,
    batch_size_sample,
    type_calibrator="iso",
):
    calib_ids = np.random.choice(np.arange(X_train.shape[0]), n_calib_pts)
    real_calib_data = [torch.FloatTensor(X_train[calib_ids])]

    BASE_D = "base"
    scores_real = {
        BASE_D: np.concatenate(
            [
                (D(data.to(device)).detach().cpu().numpy()[:, 0])
                for data in real_calib_data
            ],
        ),
    }
    scores_real_df = validate_scores(scores_real)
    n_real_batches, rem = divmod(len(scores_real[BASE_D]), batch_size_sample)
    if rem != 0:
        raise ValueError(
            "Number calibration points must be divisible by batch size",
        )

    n_dim = X_train.shape[1]

    def gen_disc_f(batch_size_fixed_):
        noise = torch.randn(batch_size_fixed_, n_dim, device=device)
        x = G(noise).detach()

        scores = {BASE_D: D(x).detach().cpu().numpy()[:, 0]}

        x = x.cpu().numpy()
        return x, scores

    _, scores_fake_df = batched_gen_and_disc(
        gen_disc_f,
        n_real_batches,
        batch_size_sample,
    )

    # outf = 'temp'
    # outf = os.path.abspath(os.path.expanduser(outf))

    # print('using dump folder:')
    # print(outf)

    # epoch = 0
    ref_method = (BASE_D, "raw")
    # incep_ref = BASE_D + '_iso_base'
    # score_fname = os.path.join(outf, '%d_scores.csv' % epoch)
    if type_calibrator == "iso":
        calib_dict = {"iso": cl.Isotonic}
    elif type_calibrator == "raw":
        calib_dict = {"raw": cl.Identity}
    elif type_calibrator == "linear":
        calib_dict = {"linear": cl.Linear}
    elif type_calibrator == "beta1":
        calib_dict = {"beta1": cl.Beta1}
    elif type_calibrator == "beta2":
        calib_dict = {"beta2": cl.Beta2}
    else:
        raise TypeError("Unknown calibrator type")

    # perf_report, calib_report, clf_df = \
    #    discriminator_analysis(scores_fake_df, scores_real_df, ref_method,
    #                           dump_fname=score_fname)
    pred_df_dump, clf_df = discriminator_analysis(
        scores_fake_df,
        scores_real_df,
        ref_method,
        calib_dict=calib_dict,
    )

    # print('image dumps...')
    # Some image dumps in case we want to actually look at generated images
    pickers = {"MH": mh.mh_sample}
    X, picked, cap_out, alpha = enhance_samples_series(
        gen_disc_f,
        scores_real_df,
        clf_df,
        pickers,
        n_samples=batch_size_sample,
    )

    return X


def enhance_samples_series_from_scratch(
    g_d_f,
    scores_real_df,
    clf_df,
    pickers,
    n_samples=16,
    batch_size=64,
    chain_batches=10,
):
    """
    Call enhance_samples multiple times to build up a batch of selected images.
    Stores list of used images X separate from the indices of the images
    selected by each method. This is more memory efficient if there are
    duplicate images selected.
    g_d_f : callable returning (X, scores) compliant with `validate`
    calibrator : dict of str -> trained sklearn classifier
        same keys as scores
    n_images : int
    """
    assert n_samples > 0
    X = []
    picked_ = None
    # picked_num = 0
    # all_generated_num = 0

    X_, scores_fake_df = batched_gen_and_disc(
        g_d_f,
        chain_batches * n_samples,
        batch_size,
    )

    for nn in tqdm(range(n_samples)):
        X_, scores_fake_df = batched_gen_and_disc(
            g_d_f,
            chain_batches,
            batch_size,
        )
        picked_, cc, aa = enhance_samples(
            scores_fake_df,
            scores_real_df,
            clf_df,
            pickers=pickers,
        )
        picked_ = picked_.unstack()  # Convert to series
        used_idx, idx_new = np.unique(picked_.values, return_inverse=True)
        add_X = list(X_[used_idx])

        X.extend(add_X)  # Unravel first index to list

    X = np.asarray(X)
    return X


@torch.no_grad()
def mh_sampling_from_scratch(
    X_train,
    G,
    D,
    device,
    n_calib_pts,
    batch_size_sample,
    n_steps,
    type_calibrator="isotonic",
    normalize_to_0_1=True,
):

    calib_ids = np.random.choice(np.arange(X_train.shape[0]), n_calib_pts)
    real_calib_data = [torch.FloatTensor(X_train[calib_ids])]

    BASE_D = "base"
    scores_real = {
        BASE_D: np.concatenate(
            [
                (D(data.to(device)).detach().cpu().numpy()[:, 0])
                for data in real_calib_data
            ],
        ),
    }
    if normalize_to_0_1:
        scores_real[BASE_D] = expit(scores_real[BASE_D])
    scores_real_df = validate_scores(scores_real)

    # scores_real_df = pd.DataFrame(scores_real)
    n_real_batches, rem = divmod(len(scores_real[BASE_D]), batch_size_sample)
    if rem != 0:
        raise ValueError(
            "Number calibration points must be divisible by batch size",
        )

    z_dim = G.z_dim

    def gen_disc_f(batch_size_fixed_):
        noise = torch.randn(batch_size_fixed_, z_dim, device=device)
        x = G(noise).detach()

        scores = {BASE_D: D(x).detach().cpu().numpy()[:, 0]}
        if normalize_to_0_1:
            scores[BASE_D] = expit(scores[BASE_D])
        x = x.cpu().numpy()
        return x, scores

    _, scores_fake_df = batched_gen_and_disc(
        gen_disc_f,
        n_real_batches,
        batch_size_sample,
    )

    min_val = min(
        np.min(scores_fake_df[BASE_D].values),
        np.min(scores_real_df[BASE_D].values),
    )

    max_val = max(
        np.max(scores_fake_df[BASE_D].values),
        np.max(scores_real_df[BASE_D].values),
    )

    if normalize_to_0_1:
        min_val = 0.0
        max_val = 1.0
    ref_method = (BASE_D, "raw")
    # incep_ref = BASE_D + '_iso_base'
    # score_fname = os.path.join(outf, '%d_scores.csv' % epoch)
    print("Used calibrator: ")
    if type_calibrator == "isotonic":
        calib_dict = {"isotonic": cl.Isotonic(min_val, max_val)}
        print(calib_dict)
    elif type_calibrator == "raw":
        calib_dict = {"raw": cl.Identity()}
        print(calib_dict)
    elif type_calibrator == "linear":
        calib_dict = {"linear": cl.Linear()}
        print(calib_dict)
    elif type_calibrator == "beta1":
        calib_dict = {"beta1": cl.Beta1()}
        print(calib_dict)
    elif type_calibrator == "beta2":
        calib_dict = {"beta2": cl.Beta2()}
        print(calib_dict)
    else:
        raise TypeError("Unknown calibrator type")

    print("start to train calibrator")
    pred_df_dump, clf_df = discriminator_analysis(
        scores_fake_df,
        scores_real_df,
        ref_method,
        calib_dict=calib_dict,
    )

    # print('image dumps...')
    # Some image dumps in case we want to actually look at generated images
    pickers = {"MH": mh.mh_sample}
    X, picked, cap_out, alpha = enhance_samples_series(
        gen_disc_f,
        scores_real_df,
        clf_df,
        pickers,
        n_samples=batch_size_sample,
        chain_batches=1,
        batch_size=n_steps,
    )

    return X
