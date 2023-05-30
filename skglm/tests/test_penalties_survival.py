from skglm.datafits import Cox_PH
from skglm.estimators import GeneralizedSurvivalEstimator
from skglm.utils.data import make_correlated_survival_data
from skglm.penalties import L0_5, L1_plus_L2, L1, L2_3, SCAD, MCPenalty
from skglm.solvers import AndersonCD


n_samples = 20
n_features = 10
n_tasks = 1


X, Y, _ = make_correlated_survival_data(
    n_samples=n_samples, n_features=n_features, density=0.5,
    random_state=0, p_censor=0.1)

n_samples, n_features = X.shape

print(Y)

tol = 1e-10

penalties = [
    L1(alpha=0.1),
    L1_plus_L2(alpha=0.1, l1_ratio=0.5),
    MCPenalty(alpha=0.1, gamma=4),
    SCAD(alpha=0.1, gamma=4),
    L0_5(0.1),
    L2_3(0.1)]


def test_cox():
    # check that when alphas = [alpha, ..., alpha], SLOPE and L1 solutions are equal
    est = GeneralizedSurvivalEstimator(
        datafit= Cox_PH(),
        penalty=L1(0.1),
        solver=AndersonCD(max_iter=1000, tol=tol),
    ).fit(X, Y)
    
    return est.coef_

test_cox()