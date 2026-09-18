"""总榜公平性的核心：双向加法劈分（天桶难度 vs 各家技巧）。

这两条性质一旦坏掉，总榜就会退化成"谁覆盖得短谁占便宜"，而页面上不会有任何
报错——所以它们必须被机器钉死：

1. **可恢复性**：数据真服从加法结构时，劈分必须精确还原各家的技巧与各档的难度。
2. **去偏性**：只覆盖容易档次的源，不得因为"白拿简单分"而排到真本事更好的源前面。
3. **降级与退化**：只有一家（无从比较）时退回各桶平均分；批量版与点估计必须逐格一致。
"""
import numpy as np

from weather_eval import stats


def test_two_way_adjust_recovers_skill_and_difficulty():
    """加法结构下精确还原行（技巧）与列（难度）效应。"""
    diff = np.array([12.0, 6.0, 0.0, -6.0, -12.0])     # 各档难度（越往后越难）
    skill = np.array([3.0, -1.0, 0.0])
    cov = np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]], dtype=bool)
    S = np.where(cov, 70.0 + skill[:, None] + diff[None, :], np.nan)
    out = stats.two_way_adjust(S, min_col=3, min_row=1)
    # 只有前两档凑满 3 家（第 3 档起家数不够，行效应与列效应分不开，剔除）
    assert out["col_keep"].tolist() == [True, True, False, False, False]
    assert out["row_keep"].tolist() == [True, True, True]
    # 行分 = 各自技巧 + 保留档的平均难度（= 70 + skill + (12+6)/2）
    assert np.allclose(out["scores"], 79.0 + skill, atol=1e-6)
    assert np.allclose(out["col_effects"][:2], diff[:2] - diff[:2].mean(), atol=1e-6)
    assert out["n_components"] == 1


def test_two_way_adjust_removes_coverage_bias():
    """覆盖越短越占便宜的偏置必须被消掉。

    三家的"真本事"依次是 C > B > A，但 A 只覆盖最容易的两档、C 覆盖全部六档。
    直接按各家自己的桶取平均（旧口径）会让 A 登顶；劈分后名次回到真本事。
    """
    diff = np.array([12.0, 6.0, 0.0, -6.0, -12.0, -18.0])
    skill = np.array([-4.0, 0.0, 4.0])                  # A 最差、C 最好
    cov = np.array([[1, 1, 0, 0, 0, 0],                 # A：只覆盖最容易的两档
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1]], dtype=bool)    # C：覆盖全部
    S = np.where(cov, 70.0 + skill[:, None] + diff[None, :], np.nan)
    raw = np.nanmean(S, axis=1)
    assert raw[0] > raw[1] > raw[2]                     # 旧口径：覆盖短的 A 反而第一
    out = stats.two_way_adjust(S, min_col=3, min_row=1)
    # 新口径：名次 = 真本事（C > B > A）
    assert out["scores"][2] > out["scores"][1] > out["scores"][0]
    assert np.allclose(out["scores"], 70.0 + skill + diff[:2].mean(), atol=1e-6)


def test_design_mask_prunes_thin_rows_and_columns():
    """同台家数不足的档、以及因此无处可比的源，都要被剔出设计。"""
    V = np.array([[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 1]], dtype=bool)   # 末列/末行只有 1 家
    rows, cols = stats.design_mask(V, min_col=3, min_row=1, rounds=6)
    assert cols.tolist() == [True, True, False, False]     # 前两列有 3 家
    # 第四行只落在被剔除的末列上 → 也不进设计
    assert rows.tolist() == [True, True, True, False]
    # 阈值放宽到 2 时第三列（2 家）进来；此时最少 2 格的要求把只有 2 格的第一行留下
    rows2, cols2 = stats.design_mask(V, min_col=2, min_row=2, rounds=6)
    assert cols2.tolist() == [True, True, True, False]
    assert rows2.tolist() == [True, True, True, False]


def test_batch_matches_point_and_degenerates_to_macro():
    """批量版与点估计逐格一致；只有一行时退化为该行的桶平均分。"""
    rng = np.random.default_rng(7)
    base = np.array([[80.0, 74.0, 68.0, np.nan],
                     [78.0, 73.0, np.nan, np.nan],
                     [76.0, 72.0, 66.0, 60.0]])
    rows, cols = stats.design_mask(np.isfinite(base), min_col=2, min_row=1)
    point = stats.two_way_adjust(base, min_col=2, min_row=1)["scores"]
    batch = stats.difficulty_adjusted(base[None, ...], rows, cols)[0]
    assert np.allclose(point, batch, atol=1e-9)
    # 多次重采样叠加噪声后，批量版仍与"逐行单独算"一致（各行互不干扰）
    S3 = np.repeat(base[None, ...], 20, axis=0) + rng.normal(0, 0.4, (20, *base.shape))
    S3 = np.where(np.isfinite(S3), S3, np.nan)
    b1 = stats.difficulty_adjusted(S3, rows, cols)
    b2 = np.stack([stats.difficulty_adjusted(S3[i:i + 1], rows, cols)[0] for i in range(20)])
    assert np.allclose(b1, b2, atol=1e-9)
    # 单行：无横向信息可劈，退化为该行的桶平均分（这也是"只接一个源"时的降级路径）
    one = np.array([[80.0, 74.0, 68.0, np.nan]])
    r1, c1 = np.isfinite(one).any(axis=1), np.isfinite(one).any(axis=0)
    assert abs(stats.difficulty_adjusted(one[None, ...], r1, c1)[0][0]
               - float(np.nanmean(one))) < 1e-9


def test_two_way_adjust_degenerate_inputs():
    """空矩阵 / 全缺 / 单个格子：不得抛异常，且给出 None 而非虚构的分数。"""
    for S in (np.full((3, 4), np.nan), np.zeros((0, 4)),
              np.array([[np.nan, 1.0], [2.0, np.nan]]), np.array([[5.0]])):
        out = stats.two_way_adjust(S, min_col=3, min_row=1)
        assert out["scores"].shape == (S.shape[0],)
        # 凑不出可比较的设计时，行分必须是 NaN（页面上显示"样本积累中"）
        if S.size and np.isfinite(S).any() and S.shape[0] > 1 and S.shape[1] > 1:
            assert np.isnan(out["scores"]).all()
