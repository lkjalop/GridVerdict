"""Quick probe to determine which MMSDM months/tables are accessible via full GET."""
import asyncio
import httpx

BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"

def price_url(year: int, month: int) -> str:
    ym = f"{year}{month:02d}"
    return (
        f"{BASE}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/"
        f"DATA/PUBLIC_DVD_DISPATCHPRICE_{ym}010000.zip"
    )

PROBES = [
    # Spot-check 2022
    ("DISPATCHPRICE_2022_01", price_url(2022, 1)),
    ("DISPATCHPRICE_2022_07", price_url(2022, 7)),
    ("DISPATCHPRICE_2022_12", price_url(2022, 12)),
    # 2023 boundary
    ("DISPATCHPRICE_2023_01", price_url(2023, 1)),
    ("DISPATCHPRICE_2023_12", price_url(2023, 12)),
    # 2024 boundary
    ("DISPATCHPRICE_2024_01", price_url(2024, 1)),
    ("DISPATCHPRICE_2024_07", price_url(2024, 7)),
    ("DISPATCHPRICE_2024_08", price_url(2024, 8)),
    # Also check constraint for 2024 (was failing before)
    ("DISPATCHCONSTRAINT_2024_07",
     f"{BASE}/2024/MMSDM_2024_07/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_DVD_DISPATCHCONSTRAINT_202407010000.zip"),
]


async def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
        for label, url in PROBES:
            try:
                async with client.stream("GET", url) as sr:
                    status = sr.status_code
                    await sr.aclose()
                marker = "OK" if status < 400 else "FAIL"
                print(f"  [{marker}] {label}: {status}")
            except Exception as exc:
                print(f"  [ERR] {label}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
