from __future__ import annotations

import gzip
import io
import zipfile
from datetime import datetime, timezone

import pytest

from app.mcp.aemo_archive import (
    _candidate_mmsdm_urls,
    parse_mmsdm_dispatchprice_content,
    parse_mmsdm_driver_content,
    parse_mmsdm_unit_content,
)


_MMSDM_CSV = """\
C,NEMP.WORLD,DISPATCH,PRICE,4
I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,RRP,TOTALDEMAND,AVAILABLEGENERATION
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,NSW1,1,0,347.50,8420.0,8850.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,VIC1,1,0,180.25,6100.0,6500.0
"""

_DRIVER_CSV = """\
C,NEMP.WORLD,DISPATCH,INTERCONNECTORRES,4
I,DISPATCH,INTERCONNECTORRES,4,SETTLEMENTDATE,RUNNO,INTERCONNECTORID,DISPATCHINTERVAL,METEREDMWFLOW,MWFLOW,EXPORTLIMIT,IMPORTLIMIT
D,DISPATCH,INTERCONNECTORRES,4,2026/05/23 14:30:00,1,N-Q-MNSP1,1,120.0,118.5,300.0,-300.0
I,DISPATCH,CONSTRAINT,4,SETTLEMENTDATE,RUNNO,CONSTRAINTID,DISPATCHINTERVAL,RHS,MARGINALVALUE,VIOLATIONDEGREE
D,DISPATCH,CONSTRAINT,4,2026/05/23 14:30:00,1,N^^Q_NIL,1,50.0,12.5,0.0
"""

_UNIT_CSV = """\
C,NEMP.WORLD,DISPATCH,UNIT_SOLUTION,4
I,DISPATCH,DISPATCH_UNIT_SOLUTION,4,SETTLEMENTDATE,RUNNO,DUID,INITIALMW,TOTALCLEARED,AVAILABILITY,RAMPUPRATE,SEMIDISPATCHCAP
D,DISPATCH,DISPATCH_UNIT_SOLUTION,4,2026/05/23 14:30:00,1,ERARING1,450.0,510.0,650.0,3.0,0
D,DISPATCH,DISPATCH_UNIT_SOLUTION,4,2026/05/23 14:30:00,1,POAT220,90.0,140.0,200.0,20.0,0
C,NEMP.WORLD,DUDETAILSUMMARY,GENUNITS,4
I,DUDETAILSUMMARY,GENUNITS,4,DUID,STATIONNAME,REGIONID,PARTICIPANTID,FUELTYPE,DISPATCHTYPE,REGISTEREDCAPACITY
D,DUDETAILSUMMARY,GENUNITS,4,ERARING1,ERARING,NSW1,ORIGIN,COAL,GENERATOR,720
"""


def test_parse_mmsdm_plain_csv():
    rows = parse_mmsdm_dispatchprice_content(_MMSDM_CSV.encode(), raw_ref="plain")
    assert {row["region"] for row in rows} == {"NSW1", "VIC1"}
    assert rows[0]["valid_time"] == datetime(2026, 5, 23, 4, 30, tzinfo=timezone.utc)
    assert rows[0]["price_rrp"] == pytest.approx(347.50)


def test_parse_mmsdm_gzip_csv():
    rows = parse_mmsdm_dispatchprice_content(gzip.compress(_MMSDM_CSV.encode()), raw_ref="gzip")
    assert len(rows) == 2
    assert rows[1]["raw_ref"] == "gzip"


def test_parse_mmsdm_zip_with_gz_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("PUBLIC_DVD_DISPATCHPRICE_202605010000.CSV.GZ", gzip.compress(_MMSDM_CSV.encode()))
    rows = parse_mmsdm_dispatchprice_content(buf.getvalue(), raw_ref="zip")
    assert len(rows) == 2
    assert all(row["source"] == "AEMO_DISPATCH_PRICE" for row in rows)


def test_parse_mmsdm_driver_content_interconnector_and_constraint():
    rows = parse_mmsdm_driver_content(_DRIVER_CSV.encode(), raw_ref="driver")
    assert {row["driver_type"] for row in rows} == {"interconnector", "constraint"}
    interconnector = next(row for row in rows if row["driver_type"] == "interconnector")
    constraint = next(row for row in rows if row["driver_type"] == "constraint")
    assert interconnector["element_id"] == "N-Q-MNSP1"
    assert interconnector["values"]["metered_mw_flow"] == pytest.approx(120.0)
    assert constraint["element_id"] == "N^^Q_NIL"
    assert constraint["values"]["marginal_value"] == pytest.approx(12.5)


def test_parse_mmsdm_unit_content_dispatch_and_metadata():
    payload = parse_mmsdm_unit_content(_UNIT_CSV.encode(), raw_ref="unit")
    assert len(payload["unit_rows"]) == 2
    assert payload["unit_rows"][0]["duid"] == "ERARING1"
    assert payload["unit_rows"][0]["fuel_type"] == "coal"
    assert payload["unit_rows"][1]["fuel_type"] == "hydro"
    assert payload["metadata_rows"][0]["duid"] == "ERARING1"
    assert payload["metadata_rows"][0]["max_capacity_mw"] == pytest.approx(720)


def test_candidate_mmsdm_urls_include_three_year_tables():
    urls = _candidate_mmsdm_urls(
        datetime(2023, 5, 1, tzinfo=timezone.utc),
        datetime(2023, 5, 1, tzinfo=timezone.utc),
        ["DISPATCHPRICE", "DISPATCHINTERCONNECTORRES", "DISPATCHCONSTRAINT", "DISPATCH_UNIT_SOLUTION"],
    )
    joined = "\n".join(urls)
    assert "PUBLIC_DVD_DISPATCHPRICE_202305010000.zip" in joined
    assert "PUBLIC_DVD_DISPATCHINTERCONNECTORRES_202305010000.zip" in joined
    assert "PUBLIC_DVD_DISPATCHCONSTRAINT_202305010000.zip" in joined
    assert "PUBLIC_DVD_DISPATCH_UNIT_SOLUTION_202305010000.zip" in joined
