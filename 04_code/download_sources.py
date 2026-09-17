"""Retrieve legally accessible Project 1 sources without overwriting raw bytes."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "01_metadata" / "source_manifest.csv"
LOG = ROOT / "09_logs" / "download_log.jsonl"
USER_AGENT = "GlobalCancerResearch-P01/0.1 (+auditable academic retrieval)"

MANIFEST_FIELDS = [
    "source_id", "source_name", "release", "local_path", "retrieval_url",
    "web_route", "accessed_at_utc", "query_parameters", "license_or_terms",
    "bytes", "sha256", "data_status", "notes",
]


DIRECT_DOWNLOAD = "direct file download; no query filters"


@dataclass(frozen=True)
class Source:
    source_id: str
    source_name: str
    release: str
    url: str
    web_route: str
    relative_path: str
    license_or_terms: str
    data_status: str
    notes: str = ""
    query_parameters: str = DIRECT_DOWNLOAD
    # Completeness check run on the retrieved bytes before they are accepted.
    # A query route can return HTTP 200 with a truncated or empty payload; the
    # raw file must never be registered as complete unless it is.
    validator: Callable[[bytes], None] | None = field(default=None, compare=False)


def check_odata_csv(expected_columns: tuple[str, ...], minimum_rows: int) -> Callable[[bytes], None]:
    def validate(payload: bytes) -> None:
        lines = payload.decode("utf-8-sig").splitlines()
        if len(lines) - 1 < minimum_rows:
            raise RuntimeError(f"Expected >= {minimum_rows} data rows, received {len(lines) - 1}")
        header = {column.strip() for column in lines[0].split(",")}
        missing = [column for column in expected_columns if column not in header]
        if missing:
            raise RuntimeError(f"Missing expected columns: {missing}")

    return validate


def check_worldbank_json(payload: bytes) -> None:
    document = json.loads(payload)
    if not isinstance(document, list) or len(document) < 2 or not document[1]:
        raise RuntimeError(f"World Bank API returned no observations: {document[:1]}")
    pages = document[0].get("pages")
    if pages != 1:
        raise RuntimeError(f"Retrieval is paginated (pages={pages}); raise per_page before accepting")


def check_gho_json(minimum_rows: int) -> Callable[[bytes], None]:
    def validate(payload: bytes) -> None:
        rows = json.loads(payload).get("value")
        if not rows or len(rows) < minimum_rows:
            raise RuntimeError(f"Expected >= {minimum_rows} GHO rows, received {len(rows or [])}")

    return validate


def worldbank_url(indicator: str) -> str:
    return (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        "?format=json&per_page=20000&date=1990:2024"
    )


def gho_url(indicator: str) -> str:
    query = urllib.parse.urlencode({"$filter": "SpatialDimType eq 'COUNTRY'"}, safe="$ '")
    return f"https://ghoapi.azureedge.net/api/{indicator}?{query}"


def wiise_url(table: str) -> str:
    query = urllib.parse.urlencode(
        {"$filter": "contains(VACCINECODE,'HPV')", "$format": "csv"}, safe="$,()' "
    )
    return f"https://xmart-api-public.who.int/WIISE/{table}?{query}"


# --- IARC Cancer Over Time ---------------------------------------------------
# The read-only JSON API that the Cancer Over Time web application itself calls.
# It is served without credentials from the base recorded in the application's
# own configuration (VUE_APP_API), and these are the same requests the interface
# issues when a user selects a site, sex and age range; nothing is scraped from
# the rendered page and no access control is involved.
#
# Path shape, read off the application's URL builder:
#   {base}/data/population/{type}/{sex}/{populations}/{cancers}/?ages_group=..
# type 0 = incidence, 1 = mortality; sex 1 = males, 2 = females (the
# application's own ``types_labels`` and ``sexes_labels``). ``ages_group`` takes
# five-year age-band indices, band i covering ages 5i to 5i+4, so 4_7 is
# 20-24 through 35-39. Cancer 16 is cervix uteri, ICD-10 C53, confirmed against
# the cancer dictionary retrieved alongside the data.
COT_API = "https://gco-api.iarc.fr/api/overtime/v2/22"
COT_CERVIX_CANCER_ID = 16
COT_FEMALE = 2
COT_INCIDENCE = 0
COT_AGE_BANDS = "4_7"
COT_YEARS = (1943, 2024)


def cancer_over_time_url(population_ids: str) -> str:
    query = urllib.parse.urlencode({
        "ages_group": COT_AGE_BANDS,
        "year_start": COT_YEARS[0],
        "year_end": COT_YEARS[1],
        "age_span": 1,
    })
    return (
        f"{COT_API}/data/population/{COT_INCIDENCE}/{COT_FEMALE}/"
        f"{population_ids}/{COT_CERVIX_CANCER_ID}/?{query}"
    )


def check_cancer_over_time_meta(minimum_rows: int, required: str) -> Callable[[bytes], None]:
    def validate(payload: bytes) -> None:
        rows = json.loads(payload)
        if not isinstance(rows, list) or len(rows) < minimum_rows:
            raise RuntimeError(
                f"Expected >= {minimum_rows} dictionary rows, received {len(rows or [])}"
            )
        if not any(required in str(row.get("label", "")) for row in rows):
            raise RuntimeError(f"Dictionary does not contain {required!r}")

    return validate


def check_cancer_over_time_data(payload: bytes) -> None:
    """Reject a partial series before it can be registered as complete.

    The endpoint answers 200 with ``{"dataset": [], "error": [...]}`` for a
    query it cannot serve, and a silently empty or single-country payload here
    would be indistinguishable from a country having no registry data.
    """
    document = json.loads(payload)
    rows = document.get("dataset")
    if not rows:
        raise RuntimeError(f"Empty dataset; API error field: {document.get('error')}")
    if document.get("error"):
        raise RuntimeError(f"API reported errors: {document['error']}")
    countries = {row["country"] for row in rows}
    ages = {row["age_label"] for row in rows}
    if len(countries) < 40:
        raise RuntimeError(f"Expected >= 40 populations, received {len(countries)}")
    if ages != {"20-24", "25-29", "30-34", "35-39"}:
        raise RuntimeError(f"Unexpected age bands: {sorted(ages)}")
    if any(row["cancer"] != COT_CERVIX_CANCER_ID or row["sex"] != COT_FEMALE for row in rows):
        raise RuntimeError("Payload contains rows outside the requested cancer/sex")


SOURCES = (
    Source(
        "who_unicef_hpv_2025",
        "WHO/UNICEF HPV immunization coverage estimates",
        "2025 revision (released 2026-07-15)",
        "https://data.unicef.org/wp-content/uploads/2025/07/hpv2025rev_web-update.xlsx",
        "https://data.unicef.org/resources/dataset/immunization/",
        "02_raw_data/who_unicef_hpv/2025_revision/hpv2025rev_web-update.xlsx",
        "CC BY 3.0 IGO stated on UNICEF dataset download page",
        "modelled",
        "Country/regional/global HPV coverage trends; preserve estimate types.",
    ),
    Source(
        "who_gho_hpv_by15_2025",
        "WHO GHO HPV immunization coverage among the primary target cohort",
        "2025 data revision (API updated 2026-07-14)",
        "https://ghoapi.azureedge.net/api/SDGHPVRECEIVED?$filter=SpatialDimType%20eq%20%27COUNTRY%27",
        "https://www.who.int/data/gho/info/gho-odata-api",
        "02_raw_data/who_unicef_hpv/2025_revision/gho_SDGHPVRECEIVED_country.json",
        "WHO GHO API terms and WHO website terms",
        "modelled",
        "Documented OData route; coverage among primary target cohort, retained as an alternative exposure series rather than silently substituted for programme final-dose coverage.",
    ),
    Source(
        "who_unicef_hpv_notes_2025",
        "WHO/UNICEF HPV coverage estimate methodology notes",
        "2025 revision (released 2026-07-15)",
        "https://cdn.who.int/media/docs/default-source/immunization/wuenic/notes_who_unicef_hpv_estimates_2026.pdf?download=true&sfvrsn=9db8d3aa_10",
        "https://www.who.int/publications/m/item/who-unicef-hpv-vaccine-coverage-estimates",
        "02_raw_data/who_unicef_hpv/2025_revision/notes_who_unicef_hpv_estimates_2026.pdf",
        "WHO website terms; citation and source acknowledgement required",
        "metadata",
        "Methodology and country-status documentation accompanying the data.",
    ),
    Source(
        "un_wpp_2024_single_age_sex_1950_2023",
        "UN World Population Prospects population by single age and sex",
        "WPP 2024 medium variant, 1950-2023",
        "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz",
        "https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format",
        "02_raw_data/un_wpp/2024_revision/WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz",
        "United Nations World Population Prospects terms and disclaimer",
        "modelled",
        "Official bulk CSV; estimates through 2023, female rows will provide the analysis offset.",
    ),
    Source(
        "iarc_cancer_over_time_populations",
        "IARC Cancer Over Time population dictionary",
        "API data version 2.2 (WIISE-independent; retrieved 2026-08-13)",
        f"{COT_API}/meta/populations/all/",
        "https://gco.iarc.who.int/overtime/en",
        "02_raw_data/iarc_cancer_over_time/2026-08-13_api/meta_populations.json",
        "Free noncommercial/nonpromotional use with appropriate reference and acknowledgement",
        "metadata",
        "Registry coverage, national/subnational flag, reporting period and registry "
        "name per population. Required to read the rates: a Cancer Over Time "
        "'country' may be one subnational registry.",
        query_parameters="path: meta/populations/all",
        validator=check_cancer_over_time_meta(60, "Denmark"),
    ),
    Source(
        "iarc_cancer_over_time_cancers",
        "IARC Cancer Over Time cancer dictionary",
        "API data version 2.2 (retrieved 2026-08-13)",
        f"{COT_API}/meta/cancers/all/",
        "https://gco.iarc.who.int/overtime/en",
        "02_raw_data/iarc_cancer_over_time/2026-08-13_api/meta_cancers.json",
        "Free noncommercial/nonpromotional use with appropriate reference and acknowledgement",
        "metadata",
        "Cancer-site dictionary. Registered so the cervix uteri site id used in "
        "the query is verifiable against its ICD-10 code rather than asserted.",
        query_parameters="path: meta/cancers/all",
        validator=check_cancer_over_time_meta(20, "Cervix uteri"),
    ),
    Source(
        "iarc_cancer_over_time_cervix_female",
        "IARC Cancer Over Time age-specific cervical cancer incidence",
        "API data version 2.2, cervix uteri (C53), female, ages 20-39, 1943-2024",
        cancer_over_time_url("all"),
        "https://gco.iarc.who.int/overtime/en",
        "02_raw_data/iarc_cancer_over_time/2026-08-13_api/"
        "cervix_female_ages20_39_annual.json",
        "Free noncommercial/nonpromotional use with appropriate reference and acknowledgement",
        "observed",
        "Annual observed age-specific incidence rates per 100,000 from population-"
        "based cancer registries. This is the longitudinal series CI5 Volume XII "
        "cannot provide: Volume XII gives one pooled period per registry, so it "
        "cannot say whether the modelled-to-observed gap trends.",
        query_parameters=(
            "type=0 (incidence); sex=2 (female); populations=all; cancer=16 "
            "(cervix uteri, C53); ages_group=4_7 (20-24 to 35-39); "
            "year_start=1943; year_end=2024; age_span=1"
        ),
        validator=check_cancer_over_time_data,
    ),
    Source(
        "iarc_ci5_xii_summary",
        "IARC Cancer Incidence in Five Continents XII summary database",
        "Volume XII (2013-2017)",
        "https://gco.iarc.who.int/media/ci5/data/vol12/Download/CI5-XII.zip",
        "https://ci5.iarc.who.int/ci5-xii/download/",
        "02_raw_data/iarc_ci5/volume_xii/CI5-XII.zip",
        "Free noncommercial/nonpromotional use with appropriate reference and acknowledgement",
        "observed",
        "Tabulated registry cases/populations, not individual records.",
    ),
    # --- WHO WIISE eJRF programme history -------------------------------------
    # The Immunization Data Portal spreadsheet export only serves 2022 onward.
    # The eJRF tables behind it carry the full 2006-2025 record needed to assign
    # cohort eligibility. Public xMart4 OData endpoints require no credentials
    # and are read-only; see https://data.who.int/about/data/whdh/xmart and
    # https://extranet.who.int/xmart4/docs/xmart_api/use_api.html.
    Source(
        "who_wiise_jrf_hpv_schedules",
        "WHO WIISE eJRF HPV vaccination schedules (AD_SCHEDULES)",
        "WIISE public OData, retrieved 2026-08-11",
        wiise_url("AD_SCHEDULES"),
        "https://immunizationdata.who.int/",
        "02_raw_data/who_wiise_jrf/2026-08-11_odata/AD_SCHEDULES_HPV.csv",
        "WHO website and data-use terms; documented public xMart4 OData endpoint",
        "country_reported",
        "Country-year target ages, schedule rounds, target population, and national/subnational scope, 2006-2025. Supersedes the 2022-2025 portal export for historical eligibility.",
        "OData: $filter=contains(VACCINECODE,'HPV'); $format=csv; no aggregation",
        check_odata_csv(("COUNTRY", "YEAR", "AGEADMINISTERED", "TARGETPOP", "GEOAREA", "SCHEDULEROUNDS"), 4000),
    ),
    Source(
        "who_wiise_jrf_hpv_introductions",
        "WHO WIISE eJRF HPV vaccine introductions (AD_VACCINE_INTRODUCTIONS)",
        "WIISE public OData, retrieved 2026-08-11",
        wiise_url("AD_VACCINE_INTRODUCTIONS"),
        "https://immunizationdata.who.int/",
        "02_raw_data/who_wiise_jrf/2026-08-11_odata/AD_VACCINE_INTRODUCTIONS_HPV.csv",
        "WHO website and data-use terms; documented public xMart4 OData endpoint",
        "country_reported",
        "Nationwide/partial introduction flags and women/special schedule fields; 1602 rows, 200 countries. Years run 2006-2029 because planned introductions are included: rows after the 2023 outcome horizon are projected, not reported, and must be excluded from eligibility onset.",
        "OData: $filter=contains(VACCINECODE,'HPV'); $format=csv; no aggregation",
        check_odata_csv(("COUNTRY", "YEAR", "NATIONWIDE", "PARTIALLY"), 1000),
    ),
    # --- Covariates -----------------------------------------------------------
    Source(
        "worldbank_gdp_pc_ppp",
        "World Bank WDI GDP per capita, PPP (constant 2021 international $)",
        "WDI current release, retrieved 2026-08-11",
        worldbank_url("NY.GDP.PCAP.PP.KD"),
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures",
        "02_raw_data/worldbank_wdi/2026-08-11/NY.GDP.PCAP.PP.KD.json",
        "World Bank Open Data terms (CC BY 4.0)",
        "derived",
        "Confounder candidate; country-year, 1990-2024. Aggregate rows are retained and must be filtered downstream.",
        "country=all; date=1990:2024; per_page=20000; format=json",
        check_worldbank_json,
    ),
    Source(
        "worldbank_urban_share",
        "World Bank WDI urban population (% of total population)",
        "WDI current release, retrieved 2026-08-11",
        worldbank_url("SP.URB.TOTL.IN.ZS"),
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures",
        "02_raw_data/worldbank_wdi/2026-08-11/SP.URB.TOTL.IN.ZS.json",
        "World Bank Open Data terms (CC BY 4.0)",
        "derived",
        "Confounder candidate; country-year, 1990-2024.",
        "country=all; date=1990:2024; per_page=20000; format=json",
        check_worldbank_json,
    ),
    Source(
        "worldbank_secondary_enrolment",
        "World Bank WDI school enrolment, secondary (% gross)",
        "WDI current release, retrieved 2026-08-11",
        worldbank_url("SE.SEC.ENRR"),
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures",
        "02_raw_data/worldbank_wdi/2026-08-11/SE.SEC.ENRR.json",
        "World Bank Open Data terms (CC BY 4.0)",
        "derived",
        "Education confounder candidate; also relevant to school-based HPV delivery. Country-year, 1990-2024, sparse.",
        "country=all; date=1990:2024; per_page=20000; format=json",
        check_worldbank_json,
    ),
    Source(
        "worldbank_hiv_prevalence_unaids",
        "World Bank WDI HIV prevalence, total (% of population ages 15-49)",
        "WDI current release, retrieved 2026-08-11",
        worldbank_url("SH.DYN.AIDS.ZS"),
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures",
        "02_raw_data/worldbank_wdi/2026-08-11/SH.DYN.AIDS.ZS.json",
        "World Bank Open Data terms (CC BY 4.0); underlying estimates UNAIDS",
        "modelled",
        "UNAIDS-sourced HIV prevalence redistributed by the World Bank. This is the documented machine route; the sex/age-disaggregated UNAIDS AIDSinfo export remains a manual step.",
        "country=all; date=1990:2024; per_page=20000; format=json",
        check_worldbank_json,
    ),
    Source(
        "worldbank_female_smoking",
        "World Bank WDI prevalence of current tobacco use, females (% of female adults)",
        "WDI current release, retrieved 2026-08-11",
        worldbank_url("SH.PRV.SMOK.FE"),
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures",
        "02_raw_data/worldbank_wdi/2026-08-11/SH.PRV.SMOK.FE.json",
        "World Bank Open Data terms (CC BY 4.0)",
        "modelled",
        "Smoking confounder candidate; WHO-modelled series redistributed by the World Bank.",
        "country=all; date=1990:2024; per_page=20000; format=json",
        check_worldbank_json,
    ),
    Source(
        "who_gho_uhc_index",
        "WHO GHO UHC Service Coverage Index (SDG 3.8.1)",
        "GHO current release, retrieved 2026-08-11",
        gho_url("UHC_INDEX_REPORTED"),
        "https://www.who.int/data/gho/info/gho-odata-api",
        "02_raw_data/who_gho/2026-08-11/UHC_INDEX_REPORTED.json",
        "WHO GHO API terms and WHO website terms",
        "modelled",
        "Replaces the World Bank UHC indicator, which the country/all route no longer serves. 195 countries, 2000-2023.",
        "OData: $filter=SpatialDimType eq 'COUNTRY'",
        check_gho_json(2000),
    ),
    Source(
        "who_gho_cervical_screening_prevalence",
        "WHO GHO cervical cancer screening prevalence, women aged 30-49 (%)",
        "GHO current release, retrieved 2026-08-11",
        gho_url("NCD_CXCA_SCREENED_WITHIN_TIMEPERIOD"),
        "https://www.who.int/data/gho/info/gho-odata-api",
        "02_raw_data/who_gho/2026-08-11/NCD_CXCA_SCREENED_WITHIN_TIMEPERIOD.json",
        "WHO GHO API terms and WHO website terms",
        "survey_estimated",
        "Single-year (2019) cross-section, 195 countries. Effect modification and sensitivity only; it is not an annual panel.",
        "OData: $filter=SpatialDimType eq 'COUNTRY'",
        check_gho_json(500),
    ),
    Source(
        "who_gho_cervical_screening_programme",
        "WHO GHO existence of a national cervical cancer screening programme",
        "GHO current release, retrieved 2026-08-11",
        gho_url("NCD_CCS_cervicalcancerscreening"),
        "https://www.who.int/data/gho/info/gho-odata-api",
        "02_raw_data/who_gho/2026-08-11/NCD_CCS_cervicalcancerscreening.json",
        "WHO GHO API terms and WHO website terms",
        "country_reported",
        "Programme existence by country-year from NCD country capacity surveys; categorical.",
        "OData: $filter=SpatialDimType eq 'COUNTRY'",
        check_gho_json(400),
    ),
    Source(
        "who_gho_cervical_screening_programme_coverage",
        "WHO GHO coverage of national cervical cancer screening programme (%)",
        "GHO current release, retrieved 2026-08-11",
        gho_url("NCD_CCS_cervicalcancerpgmcvg"),
        "https://www.who.int/data/gho/info/gho-odata-api",
        "02_raw_data/who_gho/2026-08-11/NCD_CCS_cervicalcancerpgmcvg.json",
        "WHO GHO API terms and WHO website terms",
        "country_reported",
        "Self-reported programme coverage; differs in definition from the survey-based screening prevalence series and must not be pooled with it.",
        "OData: $filter=SpatialDimType eq 'COUNTRY'",
        check_gho_json(300),
    ),
)


@dataclass(frozen=True)
class ManualSource:
    """A file obtained through an interactive route that must still be provenanced.

    Nothing is retrieved for these; the bytes already on disk are hashed and
    registered so that every file under 02_raw_data appears in the manifest.
    """

    source_id: str
    source_name: str
    release: str
    relative_path: str
    web_route: str
    query_parameters: str
    license_or_terms: str
    data_status: str
    notes: str


DASHBOARD_DIR = "02_raw_data/who_unicef_hpv/who_unicef_hpv/2026-07-22_who_hpv_dashboard"
DASHBOARD_ROUTE = (
    "https://app.powerbi.com/view?r=eyJrIjoiNDIxZTFkZGUtMDQ1Ny00MDZkLThiZDktYWFlYTdkOGU2NDcwIiwidCI6ImY2MTBjMGI3LWJkMjQtNGIzOS04MTBiLTNkYzI4MGFmYjU5MCIsImMiOjh9"
)
DASHBOARD_TERMS = "WHO website and dashboard terms"

# Five dashboard visuals export the same underlying country-profile table, so
# their bytes are identical to delivery_strategy.csv. They are registered rather
# than deleted because the raw tree must stay immutable, but they carry no
# additional information and must not be ingested as separate observations.
_DASHBOARD_DUPLICATES = (
    "hpv1_coverage.csv",
    "instroduction_status.csv",
    "introduction_year.csv",
    "schedule_stats.csv",
    "targeted_sex.csv",
)

MANUAL_SOURCES = (
    ManualSource(
        "who_hpv_dashboard_country_profile_area_2026",
        "WHO HPV dashboard country coverage series (country_profile_area visual)",
        "dashboard export saved 2026-07-22",
        f"{DASHBOARD_DIR}/country_profile_area.csv",
        DASHBOARD_ROUTE,
        "manual dashboard export; country profile area visual",
        DASHBOARD_TERMS,
        "reported",
        "Distinct schema (COUNTRYNAME, ISO_3_CODE, WHO_REGION, YEAR, ANTIGEN, COVERAGE): the only dashboard export carrying a year dimension. Previously unregistered.",
    ),
    ManualSource(
        "who_hpv_dashboard_hpvc_coverage_2026",
        "WHO HPV dashboard final-dose coverage visual (hpvc_coverage)",
        "dashboard export saved 2026-07-22",
        f"{DASHBOARD_DIR}/hpvc_coverage.csv",
        DASHBOARD_ROUTE,
        "manual dashboard export; final-dose coverage visual",
        DASHBOARD_TERMS,
        "reported",
        "Same country-profile fields as delivery_strategy.csv but with HPVC/HPV1 column order reversed, so the bytes differ. Current profile only.",
    ),
) + tuple(
    ManualSource(
        f"who_hpv_dashboard_{Path(name).stem}_2026",
        f"WHO HPV dashboard country profile visual ({Path(name).stem})",
        "dashboard export saved 2026-07-22",
        f"{DASHBOARD_DIR}/{name}",
        DASHBOARD_ROUTE,
        "manual dashboard export; 196 current country profiles",
        DASHBOARD_TERMS,
        "reported",
        "Byte-identical re-export of delivery_strategy.csv (same SHA-256). Registered for completeness; do not ingest separately.",
    )
    for name in _DASHBOARD_DUPLICATES
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_event(event: dict[str, object]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def fetch_to_temp(source: Source, directory: Path) -> tuple[Path, str]:
    """Stream a response to a temporary file, retrying transient server errors.

    Bulk query routes intermittently answer 400/429/5xx when several large
    requests are issued back to back; a permanent failure such as 403 is not
    retried because repeating it would not change the outcome.
    """
    retryable = {400, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                final_url = response.geturl()
                with tempfile.NamedTemporaryFile(delete=False, dir=directory) as tmp:
                    temp_path = Path(tmp.name)
                    shutil.copyfileobj(response, tmp)
            return temp_path, final_url
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in retryable or attempt == 3:
                raise
            time.sleep(2 ** attempt * 5)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 5)
    raise RuntimeError(f"Unreachable retry state for {source.source_id}: {last_error!r}")


def download_immutable(source: Source) -> tuple[Path, str, int, str]:
    destination = ROOT / source.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path, final_url = fetch_to_temp(source, destination.parent)

    try:
        incoming_hash = sha256(temp_path)
        incoming_size = temp_path.stat().st_size
        if incoming_size == 0:
            raise RuntimeError(f"Empty response for {source.source_id}")
        if source.validator is not None:
            # Read only for validated (query-route) sources; bulk files are unread.
            source.validator(temp_path.read_bytes())
        if destination.exists():
            existing_hash = sha256(destination)
            if existing_hash != incoming_hash:
                raise FileExistsError(
                    f"Refusing to overwrite {destination}: existing SHA-256 "
                    f"{existing_hash} differs from incoming {incoming_hash}"
                )
            temp_path.unlink()
        else:
            os.replace(temp_path, destination)
        return destination, incoming_hash, incoming_size, final_url
    finally:
        if temp_path.exists():
            temp_path.unlink()


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists() or MANIFEST.stat().st_size == 0:
        return []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def upsert_manifest(row: dict[str, object]) -> None:
    rows = read_manifest()
    key = (str(row["source_id"]), str(row["local_path"]))
    found = False
    for i, existing in enumerate(rows):
        if (existing["source_id"], existing["local_path"]) == key:
            if existing.get("sha256") and existing["sha256"] != row["sha256"]:
                raise RuntimeError(f"Manifest conflict for {key}")
            rows[i] = {field: str(row.get(field, "")) for field in MANIFEST_FIELDS}
            found = True
            break
    if not found:
        rows.append({field: str(row.get(field, "")) for field in MANIFEST_FIELDS})

    temp = MANIFEST.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, MANIFEST)


def register_manual_sources() -> int:
    """Hash and register interactively obtained files already on disk."""
    missing = 0
    for manual in MANUAL_SOURCES:
        path = ROOT / manual.relative_path
        if not path.exists():
            missing += 1
            print(f"MISSING manual source {manual.source_id}: {manual.relative_path}", file=sys.stderr)
            continue
        row = {
            "source_id": manual.source_id,
            "source_name": manual.source_name,
            "release": manual.release,
            "local_path": path.relative_to(ROOT).as_posix(),
            "retrieval_url": "",
            "web_route": manual.web_route,
            "accessed_at_utc": utc_now(),
            "query_parameters": manual.query_parameters,
            "license_or_terms": manual.license_or_terms,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "data_status": manual.data_status,
            "notes": manual.notes,
        }
        upsert_manifest(row)
        print(f"REGISTERED {manual.source_id}: {row['bytes']} bytes")
    return missing


def accept_existing_manual_copy(source: Source, error: Exception) -> bool:
    """Treat a blocked download as satisfied when the manifested file is present.

    The UNICEF workbook link answers 403 to non-interactive clients, so the file
    is obtained through the browser. Once its registered checksum still matches,
    a repeat run must not report a missing input that is in fact present.
    """
    path = ROOT / source.relative_path
    if not path.exists():
        return False
    registered = {row["local_path"]: row for row in read_manifest()}
    row = registered.get(path.relative_to(ROOT).as_posix())
    if not row or not row.get("sha256"):
        return False
    if sha256(path) != row["sha256"]:
        return False
    log_event({
        "status": "manual_copy_present", "source_id": source.source_id,
        "at_utc": utc_now(), "suppressed_error": repr(error),
        "local_path": row["local_path"], "sha256": row["sha256"],
    })
    print(f"MANUAL {source.source_id}: automated route blocked ({error}); "
          f"registered manual copy verified")
    return True


def main() -> int:
    failures = 0
    for source in SOURCES:
        started = utc_now()
        try:
            path, digest, size, final_url = download_immutable(source)
            row = {
                "source_id": source.source_id,
                "source_name": source.source_name,
                "release": source.release,
                "local_path": path.relative_to(ROOT).as_posix(),
                "retrieval_url": final_url,
                "web_route": source.web_route,
                "accessed_at_utc": started,
                "query_parameters": source.query_parameters,
                "license_or_terms": source.license_or_terms,
                "bytes": size,
                "sha256": digest,
                "data_status": source.data_status,
                "notes": source.notes,
            }
            upsert_manifest(row)
            log_event({"status": "ok", **row})
            print(f"OK {source.source_id}: {size} bytes sha256={digest}")
        except Exception as exc:  # continue to produce a complete blocker log
            if accept_existing_manual_copy(source, exc):
                continue
            failures += 1
            log_event({
                "status": "error", "source_id": source.source_id,
                "at_utc": utc_now(), "error": repr(exc),
            })
            print(f"ERROR {source.source_id}: {exc}", file=sys.stderr)

    failures += register_manual_sources()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
