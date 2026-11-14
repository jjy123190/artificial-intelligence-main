"""浏览器可操作的补充教学案例，不修改第一次原始数据。"""
from .domain import EquityCase, Entity, Holding
from .graph_io import case_to_dict


def cyclic_example():
    case = EquityCase(entities=[Entity("P1", "教学自然人", "Person"),
                               Entity("A", "循环企业甲", "Company", capital=1000),
                               Entity("B", "循环企业乙", "Company", capital=1000)],
                      holdings=[Holding("P1", "A", .8), Holding("B", "A", .2), Holding("A", "B", .5)],
                      label="循环持股权益教学案例", source="全部虚构；独立方程例子，不是真实企业记录")
    return {"case": case_to_dict(case), "target": "B", "expected_person_share": .4 / .9,
            "note": "P1对B：uB=.5uA，uA=.8+.2uB；解得44.4444%，不是只算一条路径的40%"}
