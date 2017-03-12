import numpy as np
import pandas as pd
import time
gamma = 0.00463
lam_4 = 0.0189
mu = 3.6


class SVD:
    def __init__(self, lam_1, lam_2, mu, gamma1, gamma2, user_count, item_count, f):
        self.gamma2 = gamma2
        self.gamma1 = gamma1
        self.lam_2 = lam_2
        self.lam_1 = lam_1
        self.mu = mu
        self.b_u = np.zeros(user_count)
        self.b_i = np.zeros(item_count)
        self.q = np.random.random_sample((item_count, f)) * (1 / f)
        self.p = np.random.random_sample((user_count, f)) * (1 / f)

    def predict(self, user_id, item_id) -> float:
        return self.mu \
               + self.b_i[item_id] \
               + self.b_u[user_id] \
               + sum(self.q[item_id] * self.p[user_id])

    def gradient_step(self, user_id, item_id, r) -> float:
        e_ui = r - self.predict(user_id, item_id)

        diff_b_u = self.gamma1 * (e_ui - self.lam_1 * self.b_u[user_id])
        diff_b_i = self.gamma1 * (e_ui - self.lam_1 * self.b_i[item_id])
        diff_q_i = self.gamma2 * (e_ui * self.p[user_id] - self.lam_2 * self.q[item_id])
        diff_p_u = self.gamma2 * (e_ui * self.q[item_id] - self.lam_2 * self.p[user_id])

        max_diff = max(diff_b_u, diff_b_i, np.linalg.norm(diff_q_i), np.linalg.norm(diff_p_u))

        self.b_u[user_id] += diff_b_u
        self.b_i[item_id] += diff_b_i
        self.q[item_id] += diff_q_i
        self.p[user_id] += diff_p_u

        return max_diff

    def fit(self, user_id, item_id, r, eps=0.01):
        while self.gradient_step(user_id, item_id, r) > eps:
            pass


svd = SVD(lam_4, lam_4, mu, gamma, gamma, 2649430, 17772, 100)
idx = 0
start_time = time.time()
last_time = start_time
# df = pd.DataFrame.from_csv("")
logging_interval = 100000
chunk_size = 10 ** 5
for chunk in pd.read_csv("train.csv", chunksize=chunk_size):
    for _, user, item, rating in chunk.itertuples():
        svd.gradient_step(user, item, rating)
        idx += 1
        if idx % logging_interval == 0:
            tmp_time, last_time = last_time, time.time()
            print("processed ", logging_interval, "for just: ", last_time - tmp_time)
            print("total: ", idx, "for just: ", last_time - start_time)

# submit it
submission = pd.DataFrame.from_csv("test-ids.csv")
l = []
for test_id, user, item in submission.itertuples():
    l.append([test_id, svd.predict(user, item)])

submission_file = pd.DataFrame(l, columns=["Id", "Prediction"])
submission_file.to_csv("submission.csv", index=False)
