import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import tqdm
import utils


def load(image_path):
    """
    load image
    :param image_path:  load image
    :return:            height, width, channel and of normalized images
    """
    print("Loading {}".format(image_path))
    image = utils.load_image(image_path)
    h, w, c = image.shape
    image_pixl = image.reshape((-1, c))

    # TODO: please normalize image_pixl using Z-score
    _mean = None
    _std = None
    image_norm = None

    print("Finish loading!")
    return h, w, c, image_norm


def kmeans(n_cluster, image_pixl):
    kmeans = KMeans(n_clusters=n_cluster)
    labels = kmeans.fit_predict(image_pixl)
    initial_mus = kmeans.cluster_centers_
    initial_priors, initial_covs = [], []
    for i in range(n_cluster):
        datas = image_pixl[labels == i, ...].T
        initial_covs.append(np.cov(datas))
        initial_priors.append(datas.shape[1] / len(labels))
    return initial_mus, initial_priors, initial_covs


class GMM:
    def __init__(self, ncomp, initial_mus, initial_covs, initial_priors):
        """
        :param ncomp:           the number of clusters
        :param initial_mus:     initial means
        :param initial_covs:    initial covariance matrices
        :param initial_priors:  initial mixing coefficients
        """
        self.ncomp = ncomp
        self.mus = np.asarray(initial_mus)
        self.covs = np.asarray(initial_covs)
        self.priors = np.asarray(initial_priors)

    def inference(self, datas):
        """
        E-step
        :param datas:   original data
        :return:        posterior probability (gamma) and log likelihood
        """
        probs = []
        for i in range(self.ncomp):
            mu, cov, prior = self.mus[i, :], self.covs[i, :, :], self.priors[i]
            prob = prior * multivariate_normal.pdf(datas, mean=mu, cov=cov, allow_singular=True)
            probs.append(np.expand_dims(prob, -1))
        preds = np.concatenate(probs, axis=1)

        # TODO: calc log likelihood
        log_likelihood = None

        # TODO: calc gamma
        gamma = None

        return gamma, log_likelihood

    def update(self, datas, gamma):
        """
        M-step
        :param datas:   original data
        :param gamma:    gamma
        :return:
        """
        new_mus, new_covs, new_priors = [], [], []
        soft_counts = np.sum(gamma, axis=0)
        for i in range(self.ncomp):
            # TODO: calc mu
            new_mu = None
            new_mus.append(new_mu)

            # TODO: calc cov
            new_cov = None
            new_covs.append(new_cov)

            # TODO: calc mixing coefficients
            new_prior = None
            new_priors.append(new_prior)

        self.mus = np.asarray(new_mus)
        self.covs = np.asarray(new_covs)
        self.priors = np.asarray(new_priors)

    def fit(self, data, iteration):
        prev_log_liklihood = None

        bar = tqdm.tqdm(total=iteration)
        for i in range(iteration):
            gamma, log_likelihood = self.inference(data)
            self.update(data, gamma)
            if prev_log_liklihood is not None and abs(log_likelihood - prev_log_liklihood) < 1e-10:
                break
            prev_log_likelihood = log_likelihood

            bar.update()
            bar.set_postfix({"log likelihood": log_likelihood})


def main(image_path, ncomp, iteration=500):
    ih, iw, ic, image_norm = load(image_path)

    # init mu, prior and cov
    initial_mus, initial_priors, initial_covs = kmeans(ncomp, image_norm)

    # GMM
    print("GMM begins...")
    gmm = GMM(ncomp, initial_mus, initial_covs, initial_priors)
    gmm.fit(image_norm, iteration)

    # visualize
    utils.visualize(gmm, image_norm, ncomp, ih, iw)
    print("Finish!")


if __name__ == "__main__":
    main(image_path="data/original/sample.png", ncomp=3)
