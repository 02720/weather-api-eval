"""CLI 杂项回归：monthly --month 参数校验与归档冻结保护的命令层行为。"""
import pytest

from weather_eval.__main__ import main


def test_monthly_rejects_invalid_month():
    # L7 回归：非法月份给出可读错误并以非零码退出，而非裸 traceback
    with pytest.raises(SystemExit) as ei:
        main(["monthly", "--month", "2026-13"])
    assert ei.value.code != 0
    assert "无效月份" in str(ei.value)


@pytest.mark.parametrize("bad", ["2026-1", "26-08", "2026/08", "abc", "2026-00"])
def test_monthly_rejects_malformed_months(bad):
    with pytest.raises(SystemExit):
        main(["monthly", "--month", bad])
