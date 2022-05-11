import numpy as np
import pandas as pd

from . import classification as cl


# import mlpaper.benchmark_tools as bt
# based on https://github.com/uber-research/metropolis-hastings-gans/blob/master/mhgan/classification.py with
# install mlpaper


def validate_scores(scores):
    assert isinstance(scores, dict)
    for sv in scores.values():
        assert isinstance(sv, np.ndarray)
        assert sv.dtype.kind == "f"
        assert sv.ndim == 1
        assert np.all(0 <= sv) and np.all(sv <= 1)
    scores = pd.DataFrame(scores)
    return scores


def validate_X(X):
    assert isinstance(X, np.ndarray)
    assert X.dtype.kind == "f"
    batch_size, nc, image_size, _ = X.shape
    assert X.shape == (batch_size, nc, image_size, image_size)
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


def discriminator_analysis(
    scores_fake_df,
    scores_real_df,
    ref_method,
    calib_dict,
    dump_fname=None,
    label="label",
):
    """
    scores_fake_df : DataFrame, shape (n, n_discriminators)
    scores_real_df : DataFrame, shape (n, n_discriminators)
    ref_method : (str, str)
    perf_report : str
    calib_report : str
    clf_df : DataFrame, shape (n_calibrators, n_discriminators)
    """
    # Build combined data set dataframe and train calibrators
    pred_df, y_true = cl.combine_class_df(
        neg_class_df=scores_fake_df,
        pos_class_df=scores_real_df,
    )
    pred_df, y_true, clf_df = cl.calibrate_pred_df(
        pred_df,
        y_true,
        calibrators=calib_dict,
    )
    # Make methods flat to be compatible with benchmark tools
    pred_df.columns = cl.flat_cols(pred_df.columns)
    # ref_method = cl.flat(ref_method)  # Make it flat as well

    # Do calibration analysis
    # Z = cl.calibration_diagnostic(pred_df, y_true)
    # calib_report = Z.to_string()

    # Dump prediction to csv in case we want it for later analysis
    if dump_fname is not None:
        pred_df_dump = pd.DataFrame(pred_df, copy=True)
        pred_df_dump[label] = y_true
        pred_df_dump.to_csv(dump_fname, header=True, index=False)

    # No compute report on performance of each discriminator:
    # Make it into log-scale cat distn for use with benchmark tools
    # pred_df = cl.binary_pred_to_one_hot(pred_df, epsilon=1e-12)
    # print(y_true)
    # print(pred_df)
    # perf_df, _ = btc.summary_table(pred_df, y_true,
    #                               btc.STD_CLASS_LOSS, btc.STD_BINARY_CURVES,
    #                               ref_method=ref_method)

    # crap_lim = const_dict(1)

    # try:
    #    perf_report = sp.just_format_it(perf_df, shift_mod=3,
    #                                    crap_limit_min=crap_lim,
    #                                    crap_limit_max=crap_lim,
    #                                    EB_limit=crap_lim,
    #                                    non_finite_fmt={'nan': '--'})
    # except Exception as e:
    #    print(str(e))
    #    perf_report = perf_df.to_string()
    # return perf_report, calib_report, clf_df
    return pred_df, clf_df


def base(score, score_max=None):
    """This is a normal GAN. It always just selects the first generated image
    in a series.
    """
    idx = 0
    return idx, 1.0


def enhance_samples(scores_df, scores_max, scores_real_df, clf_df, pickers):
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
            (s_max,) = calibrator.predict(np.array([scores_max[disc_name]]))
            for picker_name in sorted(pickers.keys()):
                assert isinstance(picker_name, str)
                # print(f"picker name = {picker_name}")
                idx, aa = pickers[picker_name](s_, score_max=s_max)

                if idx == 0:
                    # Try again but init from first fake
                    cap_out.loc[picker_name, (disc_name, calib_name)] = True
                    idx, aa = pickers[picker_name](s_[1:], score_max=s_max)
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
    n_images=16,
    batch_size=16,
    chain_batches=10,
    max_est_batches=156,
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

    assert n_images > 0

    _, scores_max = batched_gen_and_disc(g_d_f, max_est_batches, batch_size)
    scores_max = scores_max.max(axis=0)

    print("max scores")
    print(scores_max.to_string())

    X = []
    picked = [None] * n_images
    cap_out = [None] * n_images
    alpha = [None] * n_images
    picked_ = [None] * n_images
    picked_num = 0
    all_generated_num = 0
    for nn in range(n_images):
        X_, scores_fake_df = batched_gen_and_disc(
            g_d_f,
            chain_batches,
            batch_size,
        )
        # print(f"Shape of generated random images = {X_.shape}")
        picked_, cc, aa = enhance_samples(
            scores_fake_df,
            scores_max,
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
        print(f"number of selected images = {len(add_X)} out of {len(X_)}")

        X.extend(add_X)  # Unravel first index to list
        cap_out[nn] = cc.unstack()
        alpha[nn] = aa.unstack()

    acceptence_rate = picked_num / all_generated_num
    print(f"acceptance rate = {acceptence_rate}")
    X = np.asarray(X)
    assert X.ndim == 4
    picked = pd.concat(picked, axis=1).T
    assert picked.shape == (n_images, len(picked_))
    cap_out = pd.concat(cap_out, axis=1).T
    assert cap_out.shape == (n_images, len(picked_))
    alpha = pd.concat(alpha, axis=1).T
    assert alpha.shape == (n_images, len(picked_))
    return X, picked, cap_out, alpha
