import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
import scipy.linalg

warnings.filterwarnings("ignore")


# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
def load_and_preprocess(data_path):
    df = pd.read_csv(
        data_path, sep=',', header=0, usecols=['userId', 'movieId', 'rating'],
        dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
    )
    df.columns = ['user', 'item', 'rating']
    df = df.dropna()
    unique_users = sorted(df['user'].unique())
    unique_items = sorted(df['item'].unique())
    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx for idx, iid in enumerate(unique_items)}
    df['user_idx'] = df['user'].map(user_map)
    df['item_idx'] = df['item'].map(item_map)
    users = df['user_idx'].values.astype(np.int32)
    items = df['item_idx'].values.astype(np.int32)
    ratings = df['rating'].values.astype(np.float32)
    n_users = len(unique_users)
    n_items = len(unique_items)
    print(f"Data loaded: {len(ratings)} ratings, {n_users} users, {n_items} items")
    return ratings, users, items, n_users, n_items


# ----------------------------
# 2. 凸方法
# ----------------------------
def convex_method(ratings, users, items, n_users, n_items, r=50, tau=1.0):
    R = csr_matrix((ratings, (users, items)), shape=(n_users, n_items))
    U, s, Vt = randomized_svd(R, n_components=r, n_iter=5, random_state=42)
    s_thresh = np.maximum(s - tau, 0.0)
    return U, s_thresh, Vt


def predict_convex(U, s, Vt, test_users, test_items):
    return np.sum(U[test_users] * s * Vt[:, test_items].T, axis=1)


# ----------------------------
# 3. 非凸方法：ScaledGD
# ----------------------------
def scaledgd_method(ratings, users, items, n_users, n_items, global_mean,
                    r=50,
                    lr_initial=0.02,
                    lr_decay_rate=0.95,
                    lr_decay_step=5,
                    max_iters=80,
                    reg_l2=0.1,
                    reg_preconditioner=1e-3,
                    update_clip_value=0.1,
                    init_U=None, init_V=None,
                    batch_size=512 * 1024,
                    ema_decay=0.95):
    if init_U is not None and init_V is not None:
        print("Using spectral initialization.")
        U = init_U.copy().astype(np.float32)
        V = init_V.copy().astype(np.float32)
    else:
        print("Using random initialization.")
        np.random.seed(42)
        U = np.random.randn(n_users, r).astype(np.float32) * 0.01
        V = np.random.randn(n_items, r).astype(np.float32) * 0.01

    ratings_centered = (ratings - global_mean).astype(np.float32)
    UUt_ema = U.T @ U / n_users + reg_preconditioner * np.eye(r, dtype=np.float32)
    VVt_ema = V.T @ V / n_items + reg_preconditioner * np.eye(r, dtype=np.float32)
    lr_current = lr_initial

    for it in tqdm(range(max_iters), desc="ScaledGD"):
        if it > 0 and it % lr_decay_step == 0:
            lr_current *= lr_decay_rate

        idx = np.random.choice(len(ratings), size=min(batch_size, len(ratings)), replace=False)
        rb, cb, vb_centered = users[idx], items[idx], ratings_centered[idx]

        pred = np.sum(U[rb] * V[cb], axis=1)
        err = vb_centered - pred

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        np.add.at(grad_U, rb, err[:, None] * V[cb])
        np.add.at(grad_V, cb, err[:, None] * U[rb])
        grad_U -= reg_l2 * U
        grad_V -= reg_l2 * V

        UUt = U.T @ U / n_users
        VVt = V.T @ V / n_items
        UUt_ema = ema_decay * UUt_ema + (1 - ema_decay) * UUt + reg_preconditioner * np.eye(r, dtype=np.float32)
        VVt_ema = ema_decay * VVt_ema + (1 - ema_decay) * VVt + reg_preconditioner * np.eye(r, dtype=np.float32)

        delta_U = scipy.linalg.solve(VVt_ema, grad_U.T, assume_a='pos').T
        delta_V = scipy.linalg.solve(UUt_ema, grad_V.T, assume_a='pos').T

        update_U = np.clip(lr_current * delta_U, -update_clip_value, update_clip_value)
        update_V = np.clip(lr_current * delta_V, -update_clip_value, update_clip_value)

        U += update_U
        V += update_V

    return U, V


def predict_nonconvex(U, V, test_users, test_items, global_mean):
    return global_mean + np.sum(U[test_users] * V[test_items], axis=1)


# ----------------------------
# 4. RMSE 计算
# ----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ----------------------------
# 5. 主函数
# ----------------------------
def main(data_path, r, tau):
    print("Loading data...")
    ratings, users, items, n_users, n_items = load_and_preprocess(data_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    convex_rmses = []
    nonconvex_rmses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ratings)):
        print(f"\n=== Fold {fold + 1} ===")
        train_ratings, train_users, train_items = ratings[train_idx], users[train_idx], items[train_idx]
        test_ratings, test_users, test_items = ratings[test_idx], users[test_idx], items[test_idx]
        train_mean = np.mean(train_ratings)

        print("Running Convex Method (Randomized SVD + Soft Thresholding)...")
        train_ratings_centered = train_ratings - train_mean
        U_c, s_c, Vt_c = convex_method(train_ratings_centered, train_users, train_items, n_users, n_items, r=r, tau=tau)
        pred_c = train_mean + predict_convex(U_c, s_c, Vt_c, test_users, test_items)
        pred_c = np.clip(pred_c, 0.5, 5.0)
        rmse_c = rmse(test_ratings, pred_c)
        convex_rmses.append(rmse_c)
        print(f"Convex RMSE: {rmse_c:.4f}")

        print("Running Non-convex Method (ScaledGD) with spectral init...")
        U_init = U_c * np.sqrt(s_c)
        V_init = Vt_c.T * np.sqrt(s_c)

        U_nc, V_nc = scaledgd_method(
            train_ratings, train_users, train_items, n_users, n_items, train_mean,
            r=r, init_U=U_init, init_V=V_init
        )
        pred_nc = predict_nonconvex(U_nc, V_nc, test_users, test_items, train_mean)
        pred_nc = np.clip(pred_nc, 0.5, 5.0)
        rmse_nc = rmse(test_ratings, pred_nc)
        nonconvex_rmses.append(rmse_nc)
        print(f"ScaledGD RMSE: {rmse_nc:.4f}")

    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"Convex (Soft-thresholded SVD)   : {np.mean(convex_rmses):.4f} ± {np.std(convex_rmses):.4f}")
    print(f"Non-convex (ScaledGD)           : {np.mean(nonconvex_rmses):.4f} ± {np.std(nonconvex_rmses):.4f}")


# ----------------------------
# 6. 运行入口
# ----------------------------
if __name__ == "__main__":
    DATA_PATH = "D:/PycharmProjects/最优化2/第二次大作业/ml-20m/ratings.csv"
    RANK = 50
    TAU = 150.0


    main(data_path=DATA_PATH, r=RANK, tau=TAU)


