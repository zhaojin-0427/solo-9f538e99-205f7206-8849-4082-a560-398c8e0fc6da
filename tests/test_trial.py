import io
import itertools
import random
import re

import pytest

from app import create_app
from app.rounding import ceil_wan, floor_wan, round_wan, solve_display
from app.service import (ConflictError, analyze_trial_impact, correct_statement,
                         load_statement, lock_row, trial_lock_impact)

CSV2 = ("院系,设备,差旅\n"
        "甲,5000,5000\n"
        "乙,5000,5000\n").encode()

CSV3 = ("院系,设备,差旅\n"
        "一,4500,4500\n"
        "二,5500,5500\n"
        "三,5500,5500\n").encode()

ALT_CELL_RE = re.compile(r'data-alt-row="([^"]+)" data-alt-col="([^"]+)">(\d+)<')
ALT_ROW_RE = re.compile(r'data-alt-row-subtotal="([^"]+)">(\d+)<')
ALT_COL_RE = re.compile(r'data-alt-col-subtotal="([^"]+)">(\d+)<')
ALT_TOTAL_RE = re.compile(r'data-alt-grand-total>(\d+)<')


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.sqlite3")


def _upload(db_path, csv_bytes):
    app = create_app(db_path)
    with app.test_client() as c:
        c.post("/upload", data={"csv": (io.BytesIO(csv_bytes), "a.csv")},
               content_type="multipart/form-data")
    return app


def _client(db_path, csv_bytes):
    app = _upload(db_path, csv_bytes)
    return app.test_client()


def assert_solution_valid(cells, locks, sol):
    """独立复核一个公示表解：取整合法、横竖加平、锁定行保持原值。"""
    R, C = len(cells), len(cells[0])
    P = sol["P"]
    rows = [sum(r) for r in P]
    cols = [sum(P[i][j] for i in range(R)) for j in range(C)]
    assert sol["rows"] == rows
    assert sol["cols"] == cols
    assert sum(rows) == sol["total"] == round_wan(sum(map(sum, cells)))
    locks = locks or {}
    for i in range(R):
        if i in locks:
            assert (rows[i], P[i]) == (locks[i][0], list(locks[i][1]))
        else:
            exact = sum(cells[i])
            assert floor_wan(exact) <= rows[i] <= ceil_wan(exact)
            for j in range(C):
                v = cells[i][j]
                allowed = {v // 10000} if v % 10000 == 0 else {v // 10000, v // 10000 + 1}
                assert P[i][j] in allowed
    for j in range(C):
        exact = sum(cells[i][j] for i in range(R))
        assert floor_wan(exact) <= cols[j] <= ceil_wan(exact)


def feasible_value_sets(cells, locks):
    """暴力枚举所有可行公示表，返回每格能取的值集合。"""
    locks = locks or {}
    R, C = len(cells), len(cells[0])
    options = []
    for i in range(R):
        for j in range(C):
            v = cells[i][j]
            opts = {v // 10000}
            if v % 10000:
                opts.add(v // 10000 + 1)
            options.append(sorted(opts))
    total = round_wan(sum(map(sum, cells)))
    values = {(i, j): set() for i in range(R) for j in range(C)}
    for combo in itertools.product(*options):
        P = [list(combo[i * C:(i + 1) * C]) for i in range(R)]
        rows = [sum(r) for r in P]
        cols = [sum(P[i][j] for i in range(R)) for j in range(C)]
        if sum(rows) != total:
            continue
        ok = True
        for i in range(R):
            if i in locks:
                if (rows[i], P[i]) != (locks[i][0], list(locks[i][1])):
                    ok = False
                    break
            else:
                exact = sum(cells[i])
                if not (floor_wan(exact) <= rows[i] <= ceil_wan(exact)):
                    ok = False
                    break
        if not ok:
            continue
        for j in range(C):
            exact = sum(cells[i][j] for i in range(R))
            if not (floor_wan(exact) <= cols[j] <= ceil_wan(exact)):
                ok = False
                break
        if not ok:
            continue
        for i in range(R):
            for j in range(C):
                values[(i, j)].add(P[i][j])
    return values


# ---------- 求解器 forced 参数 ----------

def test_forced_pins_cell_value():
    cells = [[5000, 5000], [5000, 5000]]
    sol = solve_display(cells, forced={(0, 0): 1})
    assert sol["P"][0][0] == 1
    assert_solution_valid(cells, {}, sol)
    sol = solve_display(cells, forced={(1, 1): 0})
    assert sol["P"][1][1] == 0
    assert_solution_valid(cells, {}, sol)


def test_forced_rejects_illegal_values():
    cells = [[5000, 5000], [5000, 5000]]
    assert solve_display(cells, forced={(0, 0): 2}) is None      # 超出上/下取整
    assert solve_display(cells, forced={(0, 0): -1}) is None     # 负数
    assert solve_display(cells, forced={(2, 0): 0}) is None      # 行越界
    assert solve_display(cells, forced={(0, 2): 0}) is None      # 列越界
    assert solve_display(cells, forced={(0.0, 0): 0}) is None    # 非整数下标
    assert solve_display(cells, forced={"x": 0}) is None         # 非法键
    # 整万格取整唯一
    assert solve_display([[10000]], forced={(0, 0): 1}) is not None
    assert solve_display([[10000]], forced={(0, 0): 0}) is None
    assert solve_display([[10000]], forced={(0, 0): 2}) is None


def test_forced_on_locked_row_must_match_lock():
    cells = [[5000, 5000], [5000, 5000]]
    base = solve_display(cells)
    locks = {0: (base["rows"][0], base["P"][0][:])}
    v = base["P"][0][0]
    sol = solve_display(cells, locks, forced={(0, 0): v})
    assert sol is not None
    assert_solution_valid(cells, locks, sol)
    assert solve_display(cells, locks, forced={(0, 0): v + 1}) is None


def test_forced_infeasible_returns_none():
    # 两列小计都恰为 1：强制 (0,0) 与 (0,1) 同时向上会让第一行小计超限
    cells = [[5000, 5000], [5000, 5000]]
    assert solve_display(cells, forced={(0, 0): 1, (0, 1): 1}) is None


# ---------- 纯函数 analyze_trial_impact 与暴力枚举比对 ----------

def test_analyze_matches_brute_force_random():
    rng = random.Random(11)
    vals = [0, 2500, 4500, 5000, 5500, 7500, 9999, 10000, 10001]
    for R, C in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        for _ in range(60):
            cells = [[rng.choice(vals) for _ in range(C)] for _ in range(R)]
            sol0 = solve_display(cells)
            assert sol0 is not None
            locks = {}
            if rng.random() < 0.5 and R > 1:
                i = rng.randrange(R)
                locks = {i: (sol0["rows"][i], sol0["P"][i][:])}
                sol0 = solve_display(cells, locks)
                if sol0 is None:
                    continue
            unlocked = [i for i in range(R) if i not in locks]
            trial_row = rng.choice(unlocked)
            trial_lock = (sol0["rows"][trial_row], sol0["P"][trial_row][:])
            result = analyze_trial_impact(cells, locks, trial_row, trial_lock)

            before = feasible_value_sets(cells, locks)
            trial_locks = dict(locks)
            trial_locks[trial_row] = trial_lock
            after = feasible_value_sets(cells, trial_locks)
            expected = {(i, j) for (i, j), vs in before.items()
                        if i not in trial_locks and len(vs) == 2
                        and len(after[(i, j)]) == 1}
            got = {(i, j) for i, j, _ in result["affected"]}
            assert got == expected, (cells, locks, trial_row, result["affected"])
            for i, j, unique in result["affected"]:
                assert after[(i, j)] == {unique}
                assert before[(i, j)] == {cells[i][j] // 10000, cells[i][j] // 10000 + 1}
            if expected:
                alt = result["alt"]
                assert alt is not None
                i, j = alt["picked"]
                assert (i, j) in expected
                assert alt["lost"] in before[(i, j)]
                assert alt["lost"] not in after[(i, j)]
                sol = alt["solution"]
                assert sol["P"][i][j] == alt["lost"]
                # 示例表是试锁前的完整可行解：取整、横竖加平、已有锁定都满足
                assert_solution_valid(cells, locks, sol)
            else:
                assert result["alt"] is None


def test_analyze_two_by_two_handcheck():
    """2×2 每格 5000：行列小计都恰为 1，可行表只有两张置换表。
    试锁任一行后，另一行两格都被唯一确定。"""
    cells = [[5000, 5000], [5000, 5000]]
    sol0 = solve_display(cells)
    trial_lock = (sol0["rows"][0], sol0["P"][0][:])
    result = analyze_trial_impact(cells, {}, 0, trial_lock)
    assert sorted(result["affected"]) == [(1, 0, 1 - sol0["P"][0][0]),
                                          (1, 1, 1 - sol0["P"][0][1])]
    alt = result["alt"]
    assert alt["picked"] == (1, 0)
    assert alt["lost"] == sol0["P"][0][0]  # 另一值恰为试锁行该列的所见值
    assert alt["solution"]["P"][1][0] == alt["lost"]
    assert_solution_valid(cells, {}, alt["solution"])


# ---------- 服务层 trial_lock_impact ----------

def test_trial_lock_impact_two_by_two(db_path):
    _upload(db_path, CSV2)
    stmt = load_statement(db_path)
    result = trial_lock_impact(db_path, "r1", stmt["version"])
    assert result["row_label"] == "甲"
    assert result["trial_cells"] == [stmt["public"][("r1", "c1")],
                                     stmt["public"][("r1", "c2")]]
    assert result["trial_subtotal"] == stmt["row_totals"]["r1"]
    assert len(result["affected"]) == 2
    by_col = {a["col_key"]: a for a in result["affected"]}
    assert set(by_col) == {"c1", "c2"}
    for col_key, a in by_col.items():
        assert a["row_key"] == "r2" and a["row_label"] == "乙"
        assert a["floor"] == 0 and a["ceil"] == 1
        # 列小计恰为 1：试锁行该列取 1 则乙只能取 0，反之亦然
        assert a["unique"] == 1 - stmt["public"][("r1", col_key)]
    alt = result["alt"]
    assert alt["row_label"] == "乙"
    assert alt["lost"] == 1 - by_col[alt["col_key"]]["unique"]
    sol = alt["solution"]
    picked_j = 0 if alt["col_key"] == "c1" else 1
    assert sol["P"][1][picked_j] == alt["lost"]
    assert_solution_valid(
        [[5000, 5000], [5000, 5000]], {}, sol)
    # 试锁不改表：版本、底表、公示表、锁定全部不变
    after = load_statement(db_path)
    assert after["version"] == stmt["version"]
    assert after["base_version"] == stmt["base_version"]
    assert not after["locks"]
    assert dict(after["public"]) == dict(stmt["public"])
    assert dict(after["base"]) == dict(stmt["base"])


def test_trial_lock_respects_existing_locks(db_path):
    _upload(db_path, CSV3)
    stmt = load_statement(db_path)
    lock_row(db_path, "r1", stmt["version"])
    stmt = load_statement(db_path)
    result = trial_lock_impact(db_path, "r2", stmt["version"])
    # 已锁定行不出现在受影响清单，试锁行本身也不出现
    assert all(a["row_key"] not in ("r1", "r2") for a in result["affected"])
    assert result["locked_keys"] == {"r1"}
    if result["alt"] is not None:
        lock = stmt["locks"]["r1"]
        assert result["alt"]["solution"]["P"][0] == lock["cells"]
        assert result["alt"]["solution"]["rows"][0] == lock["subtotal"]


def test_trial_lock_no_other_unlocked_rows(db_path):
    _upload(db_path, CSV2)
    stmt = load_statement(db_path)
    lock_row(db_path, "r2", stmt["version"])
    stmt = load_statement(db_path)
    result = trial_lock_impact(db_path, "r1", stmt["version"])
    assert result["affected"] == []
    assert result["alt"] is None


def test_trial_lock_rejects_locked_or_missing_row(db_path):
    _upload(db_path, CSV2)
    stmt = load_statement(db_path)
    lock_row(db_path, "r1", stmt["version"])
    stmt = load_statement(db_path)
    with pytest.raises(ConflictError):
        trial_lock_impact(db_path, "r1", stmt["version"])
    from app.service import ValidationError
    with pytest.raises(ValidationError):
        trial_lock_impact(db_path, "r99", stmt["version"])


def test_trial_lock_rejects_stale_version(db_path):
    _upload(db_path, CSV2)
    stmt = load_statement(db_path)
    lock_row(db_path, "r2", stmt["version"])  # 版号 +1
    with pytest.raises(ConflictError) as err:
        trial_lock_impact(db_path, "r1", stmt["version"])
    assert "请刷新" in str(err.value)


def test_trial_lock_detects_lock_during_computation(db_path):
    """核算期间另一请求锁定院系：锁定正常落库，试锁只提示刷新。"""
    _upload(db_path, CSV3)
    stmt = load_statement(db_path)

    def compute(cells, locks, forced=None):
        if not getattr(compute, "done", False):
            compute.done = True
            lock_row(db_path, "r2", load_statement(db_path)["version"])
        return solve_display(cells, locks, forced)

    with pytest.raises(ConflictError) as err:
        trial_lock_impact(db_path, "r1", stmt["version"], compute=compute)
    assert "请刷新" in str(err.value)
    new = load_statement(db_path)
    assert "r2" in new["locks"]  # 核算期间仍可锁定
    assert new["version"] == stmt["version"] + 1


def test_trial_lock_detects_correction_during_computation(db_path):
    """核算期间另一请求更正底表：更正正常生效，试锁只提示刷新。"""
    _upload(db_path, CSV3)
    stmt = load_statement(db_path)

    def compute(cells, locks, forced=None):
        if not getattr(compute, "done", False):
            compute.done = True
            new_cells = [row[:] for row in cells]
            new_cells[0][0] = 6000
            correct_statement(db_path, new_cells, stmt["base_version"])
        return solve_display(cells, locks, forced)

    with pytest.raises(ConflictError) as err:
        trial_lock_impact(db_path, "r1", stmt["version"], compute=compute)
    assert "请刷新" in str(err.value)
    new = load_statement(db_path)
    assert new["base"][("r1", "c1")] == 6000  # 核算期间仍可更正
    assert new["verified"] is True


# ---------- 路由 ----------

def test_trial_lock_route_shows_affected_and_alt_table(db_path):
    client = _client(db_path, CSV2)
    stmt = load_statement(db_path)
    resp = client.post("/trial-lock/r1", data={"version": stmt["version"]})
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "试锁影响分析：甲" in html
    assert "失去取整余地的格（2 个）" in html
    assert "乙" in html and "唯一值" in html
    assert "未改动公示表，也未保存锁定" in html
    # 示例表按页面真实数字复算横竖加平
    cells = {(m.group(1), m.group(2)): int(m.group(3)) for m in ALT_CELL_RE.finditer(html)}
    row_sub = {m.group(1): int(m.group(2)) for m in ALT_ROW_RE.finditer(html)}
    col_sub = {m.group(1): int(m.group(2)) for m in ALT_COL_RE.finditer(html)}
    total = int(ALT_TOTAL_RE.search(html).group(1))
    assert len(cells) == 4
    for r in ("r1", "r2"):
        assert sum(cells[(r, c)] for c in ("c1", "c2")) == row_sub[r]
    for c in ("c1", "c2"):
        assert sum(cells[(r, c)] for r in ("r1", "r2")) == col_sub[c]
    assert sum(row_sub.values()) == total == sum(col_sub.values())
    assert total == 2
    # 试锁不改表
    after = load_statement(db_path)
    assert after["version"] == stmt["version"] and not after["locks"]


def test_trial_lock_route_no_impact_message(db_path):
    client = _client(db_path, CSV2)
    stmt = load_statement(db_path)
    client.post("/lock/r2", data={"version": stmt["version"]})
    stmt = load_statement(db_path)
    resp = client.post("/trial-lock/r1", data={"version": stmt["version"]})
    html = resp.get_data(as_text=True)
    assert "不会减少其他未锁定格的取整选择" in html
    assert "失去取整余地的格" not in html


def test_trial_lock_route_stale_version_only_prompts_refresh(db_path):
    client = _client(db_path, CSV2)
    stmt = load_statement(db_path)
    lock_row(db_path, "r2", stmt["version"])
    resp = client.post("/trial-lock/r1", data={"version": stmt["version"]},
                       follow_redirects=True)
    html = resp.get_data(as_text=True)
    assert "请刷新" in html
    assert "试锁影响分析" not in html  # 只提示刷新，不展示过时结果


def test_trial_lock_route_version_change_during_computation(db_path, monkeypatch):
    """路由层：核算期间版号变化 → 只提示刷新；期间的锁定正常保存。"""
    client = _client(db_path, CSV3)
    stmt = load_statement(db_path)
    from app import web
    from app.service import trial_lock_impact as real_impact

    def compute(cells, locks, forced=None):
        if not getattr(compute, "done", False):
            compute.done = True
            lock_row(db_path, "r2", load_statement(db_path)["version"])
        return solve_display(cells, locks, forced)

    def patched(db, row_key, version):
        return real_impact(db, row_key, version, compute=compute)

    monkeypatch.setattr(web, "trial_lock_impact", patched)
    resp = client.post("/trial-lock/r1", data={"version": stmt["version"]},
                       follow_redirects=True)
    html = resp.get_data(as_text=True)
    assert "请刷新" in html
    assert "试锁影响分析" not in html
    assert "r2" in load_statement(db_path)["locks"]


def test_trial_lock_button_on_statement_page(db_path):
    client = _client(db_path, CSV2)
    stmt = load_statement(db_path)
    html = client.get("/statement").get_data(as_text=True)
    assert 'action="/trial-lock/r1"' in html
    assert 'action="/trial-lock/r2"' in html
    client.post("/lock/r1", data={"version": stmt["version"]})
    html = client.get("/statement").get_data(as_text=True)
    assert 'action="/trial-lock/r1"' not in html  # 已锁定行不再显示试锁
    assert 'action="/trial-lock/r2"' in html
