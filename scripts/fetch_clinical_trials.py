import os
import requests
import json
import logging
import csv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ClinicalTrials.gov API v2 base URL
API_URL = "https://clinicaltrials.gov/api/v2/studies"

# ✅ Search term (modify for different conditions)
SEARCH_TERM = "Cancer"

# ✅ Define fields to extract
FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.designModule.enrollmentInfo.count"
]

# ✅ Define the CSV file path
CSV_FILE = Path("../data/corpus/clinical_trials_corpus.csv")

def fetch_trials(page_size=100, max_pages=5):
    """Fetches clinical trials from ClinicalTrials.gov API and saves them to CSV."""
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True) 
    file_exists = CSV_FILE.exists() # pathlib version

    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # ✅ Write CSV header
        writer.writerow(["Trial ID", "Title", "Condition", "Intervention", "Study Type", "Phase", "Summary", "Eligibility", "Enrollment"])

        next_page_token = None  # Initialize the token for pagination

        for page in range(1, max_pages + 1):
            params = {
                "query.titles": SEARCH_TERM,
                "fields": ",".join(FIELDS),
                "pageSize": page_size,
                "format": "json",
            }

            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                response = requests.get(API_URL, params=params, timeout=10)
                response.raise_for_status()  # Raise HTTP error if occurs
                data = response.json()

                next_page_token = data.get("nextPageToken", None)
                studies = data.get("studies", [])

                if not studies:
                    logging.warning(f"No trials found on page {page}. Stopping pagination.")
                    break

                for study in studies:
                    protocol = study.get("protocolSection", {})

                    trial_id = protocol.get("identificationModule", {}).get("nctId", "N/A")
                    title = protocol.get("identificationModule", {}).get("briefTitle", "N/A")
                    conditions = protocol.get("conditionsModule", {}).get("conditions", ["N/A"])
                    study_type = protocol.get("designModule", {}).get("studyType", "N/A")
                    phases = protocol.get("designModule", {}).get("phases", ["N/A"])
                    summary = protocol.get("descriptionModule", {}).get("briefSummary", "N/A")
                    eligibility = protocol.get("eligibilityModule", {}).get("eligibilityCriteria", "N/A")
                    enrollment = protocol.get("designModule", {}).get("enrollmentInfo", {}).get("count", "N/A")

                    # ✅ Extract interventions safely
                    interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
                    intervention_names = [intervention.get("name", "N/A") for intervention in interventions]

                    row = [
                        trial_id,
                        title,
                        ", ".join(conditions),
                        ", ".join(intervention_names) if intervention_names else "N/A",
                        study_type,
                        ", ".join(phases),
                        summary,
                        eligibility,
                        enrollment
                    ]

                    writer.writerow(row)
                    logging.info(f"Saved: {trial_id}")

                if not next_page_token:
                    logging.info("No more pages to fetch. Stopping.")
                    break

            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP request failed on page {page}: {e}")
                break

            except json.JSONDecodeError:
                logging.error(f"Failed to parse API response as JSON on page {page}")
                break

            except Exception as e:
                logging.error(f"Unexpected error on page {page}: {e}")
                break

if __name__ == "__main__":
    fetch_trials()
