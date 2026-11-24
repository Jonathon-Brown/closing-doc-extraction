#!/usr/bin/env python3
"""
build_fl_corpus.py - Assemble a Florida real-estate CLOSING-DOCUMENT demo corpus
under the working directory (see paths.py).

Hybrid, and fully transparent about it:
  * REAL downloads  - genuine public PDFs (FAR/BAR AS-IS contract + CR-6 riders
    from Florida Realtors; CFPB TRID / Closing Disclosure samples). Filenames
    preserved from the source. origin = "real-download", with source_url.
  * SYNTHETIC docs  - realistic Orange County, FL recorded deeds, mortgages,
    closing disclosures, title commitments, and condo/HOA disclosures generated
    locally with FAKE names/parcels/book-page/instrument numbers. The county
    records portal has no API / bulk download, so these stand in for volume.
    origin = "synthetic".

Every file is labeled real vs synthetic in manifest.csv. All synthetic data is
fabricated; no real private data is used. Target 150-400 files.

Rerun-safe: wipes corpus/ and the manifest and rebuilds.
"""
from __future__ import annotations

import csv
import random
import shutil
import urllib.request
from pathlib import Path

from docx import Document
from fpdf import FPDF
from fpdf.enums import XPos, YPos

random.seed(20260828)

from paths import WORKDIR as ROOT, CORPUS
MANIFEST = ROOT / "manifest.csv"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"

rows = []   # (filename, type, source, date, origin, source_url)


# ---- REAL downloads ---------------------------------------------------------
# (url, preferred_filename, doc_type, source_label)
REAL = [
    ("https://www.floridarealtors.org/sites/default/files/2026-02/AS%20IS%20Residential%20Contract%20for%20Sale%20and%20Purchase%20%28FloridaRealtors-FloridaBar-ASIS-7x%29_Redlined%5B1%5D.pdf",
     "FloridaRealtors-FloridaBar_AS-IS_Residential_Contract_ASIS-7x.pdf",
     "AS-IS purchase contract", "Florida Realtors / Florida Bar"),
    ("https://www.floridarealtors.org/sites/default/files/2024-09/CR-6_FF.%20CREDIT%20RELATED%20TO%20BUYER%E2%80%99S%20BROKER%20COMPENSATION.pdf",
     "CR-6_FF_Credit_Related_to_Buyers_Broker_Compensation.pdf",
     "CR-6 rider/addendum", "Florida Realtors / Florida Bar"),
    ("https://www.floridarealtors.org/sites/default/files/2024-09/CR-6_GG.%20SELLER%E2%80%99S%20AGREEMENT%20WITH%20RESPECT%20TO%20BUYER%E2%80%99S%20BROKER%20COMPENSATION.pdf",
     "CR-6_GG_Sellers_Agreement_re_Buyers_Broker_Compensation.pdf",
     "CR-6 rider/addendum", "Florida Realtors / Florida Bar"),
    ("https://files.consumerfinance.gov/f/201311_cfpb_kbyo_closing-disclosure.pdf",
     "CFPB_sample_Closing_Disclosure_H-25.pdf",
     "Closing disclosure (sample)", "CFPB (federal, public domain)"),
    ("https://files.consumerfinance.gov/f/201311_cfpb_kbyo_closing-disclosure_blank.pdf",
     "CFPB_blank_Closing_Disclosure_H-25.pdf",
     "Closing disclosure (blank)", "CFPB (federal, public domain)"),
    ("https://files.consumerfinance.gov/f/documents/cfpb_pace_closing-disclosure-form-blank.pdf",
     "CFPB_blank_PACE_Closing_Disclosure_H-25K.pdf",
     "Closing disclosure (blank)", "CFPB (federal, public domain)"),
    ("https://files.consumerfinance.gov/f/201506_cfpb_closing-disclosure_cover-H25HI.pdf",
     "CFPB_Closing_Disclosure_cover_H-25HI.pdf",
     "Closing disclosure (sample)", "CFPB (federal, public domain)"),
    ("https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25C.pdf",
     "CFPB_Closing_Disclosure_cover_H-25C.pdf",
     "Closing disclosure (sample)", "CFPB (federal, public domain)"),
    ("https://files.consumerfinance.gov/f/documents/201707_cfpb_KBYO_Guide-Loan-Estimate-and-Closing-Disclosure-forms.pdf",
     "CFPB_Guide_to_Loan-Estimate_and_Closing-Disclosure_forms.pdf",
     "TRID guide", "CFPB (federal, public domain)"),
    ("https://files.consumerfinance.gov/f/201503_cfpb_tila-respa-integrated-disclosure-guide-to-the-loan-estimate-and-closing.pdf",
     "CFPB_TILA-RESPA_Integrated_Disclosure_Guide.pdf",
     "TRID guide", "CFPB (federal, public domain)"),
]


def download_real():
    import subprocess
    dest = CORPUS / "real-forms"
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for url, fname, dtype, source in REAL:
        out = dest / fname
        # curl with browser-ish headers; CFPB's CDN 403s bare urllib requests.
        ref = "https://www.consumerfinance.gov/" if "consumerfinance" in url \
              else "https://www.floridarealtors.org/"
        try:
            subprocess.run(
                ["curl", "-sL", "-A", UA, "-H", "Accept: application/pdf,*/*",
                 "-H", f"Referer: {ref}", "-o", str(out), url],
                check=True, timeout=90)
            data = out.read_bytes() if out.exists() else b""
            if data[:4] != b"%PDF":
                print(f"  SKIP (not a PDF): {fname}")
                out.unlink(missing_ok=True)
                continue
            rows.append((fname, dtype, source, "", "real-download", url))
            ok += 1
            print(f"  real  {fname}  ({len(data)//1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {fname}: {e}")
    return ok


# ---- Synthetic FL closing docs ----------------------------------------------
CITIES = ["Orlando", "Winter Park", "Apopka", "Ocoee", "Winter Garden", "Maitland",
          "Windermere", "Belle Isle", "Edgewood", "Gotha", "Zellwood", "Oakland"]
SUBS = ["Vista Lakes", "Stoneybrook", "Baldwin Park", "Avalon Park", "Lake Nona",
        "Hunters Creek", "Dr Phillips", "Meadow Woods", "Wedgefield", "Rio Pinar",
        "Waterford Lakes", "Southchase", "Timber Springs", "Eastwood"]
FIRST = ["James", "Maria", "Robert", "Linda", "Michael", "Patricia", "David", "Jennifer",
         "William", "Elizabeth", "Carlos", "Sofia", "Andre", "Nadia", "Kevin", "Grace"]
LAST = ["Thompson", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
        "Lopez", "Wilson", "Anderson", "Nguyen", "Patel", "Okafor", "Brooks", "Reyes"]
LENDERS = ["SunState Mortgage LLC", "Central Florida Federal Credit Union",
           "Orange Blossom Bank, N.A.", "Meridian Home Lending", "First Coast Savings",
           "Everglades Mortgage Corp", "Citrus Community Bank"]
TITLE_COS = ["Sunshine Title & Escrow", "Orange County Title Partners",
             "First Meridian Title Co", "Lakeside Title Services"]


def person():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def parcel():
    return (f"{random.randint(10,35)}-{random.randint(20,29)}-{random.randint(10,31)}-"
            f"{random.randint(1000,9999)}-{random.randint(10,99)}-{random.randint(100,999)}")


def legal_desc():
    lot = random.randint(1, 240); block = random.randint(1, 40)
    sub = random.choice(SUBS)
    pb = random.randint(10, 120); pg = random.randint(1, 150)
    return (f"Lot {lot}, Block {block}, {sub}, according to the plat thereof as "
            f"recorded in Plat Book {pb}, Page {pg}, of the Public Records of "
            f"Orange County, Florida.")


def rec_info(i):
    book = 20000 + i // 3
    page = (i * 37) % 9000 + 100
    instr = 20250000000 + i
    date = f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
    return book, page, instr, date


def prop_addr():
    n = random.randint(100, 9999)
    st = random.choice(["Oak St", "Palmetto Ave", "Lakeview Dr", "Magnolia Blvd",
                        "Cypress Ln", "Semoran Blvd", "Conway Rd", "Dean Rd",
                        "Curry Ford Rd", "Alafaya Trl", "Hiawassee Rd"])
    city = random.choice(CITIES)
    z = random.randint(32801, 32837)
    return f"{n} {st}, {city}, FL {z}"


def pdf_doc(path: Path, title: str, lines: list[str]):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(pdf.epw, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2); pdf.set_font("Helvetica", "", 10)
    for ln in lines:
        if ln.strip() == "":
            pdf.ln(4)
        elif ln.startswith("## "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(pdf.epw, 6, ln[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.multi_cell(pdf.epw, 5, ln, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(path))


def add(subdir, filename, dtype, date, writer):
    d = CORPUS / subdir
    d.mkdir(parents=True, exist_ok=True)
    writer(d / filename)
    rows.append((filename, dtype, "Orange County, FL (synthetic)", date, "synthetic", ""))


def gen_deed(i):
    book, page, instr, date = rec_info(i)
    grantor = person(); grantee = person()
    consideration = random.randint(180, 850) * 1000
    doc_stamps = round(consideration * 0.007, 2)
    kind = random.choice(["Warranty Deed", "Special Warranty Deed", "Quitclaim Deed"])
    lines = [
        f"Prepared by / Return to: {random.choice(TITLE_COS)}, Orlando, FL",
        f"Parcel Identification No.: {parcel()}",
        f"Recorded {date}  OR Book {book}, Page {page}  Instrument #{instr}",
        f"Documentary Stamp Tax: ${doc_stamps:,.2f}", "",
        f"## {kind}",
        f"THIS {kind.upper()}, made this {date}, by {grantor} ('Grantor'), whose "
        f"address is {prop_addr()}, to {grantee} ('Grantee').",
        f"WITNESSETH, that the Grantor, for and in consideration of the sum of "
        f"${consideration:,}.00 and other good and valuable consideration, the "
        f"receipt whereof is hereby acknowledged, has granted, bargained and sold "
        f"to the Grantee the following described land in Orange County, Florida:",
        legal_desc(),
        f"Property address: {prop_addr()}.", "",
        "IN WITNESS WHEREOF, Grantor has hereunto set Grantor's hand and seal the "
        "day and year first above written.",
        f"Signed: {grantor}    Witnesses: {person()}, {person()}",
        f"State of Florida, County of Orange. Sworn before {person()}, Notary Public.",
    ]
    fn = f"{kind.replace(' ','_')}_ORBook-{book}_Page-{page}_INSTR-{instr}.pdf"
    add("deeds", fn, kind, date, lambda p: pdf_doc(p, kind, lines))


def gen_mortgage(i):
    book, page, instr, date = rec_info(i + 5000)
    borrower = person(); lender = random.choice(LENDERS)
    principal = random.randint(150, 780) * 1000
    intangible = round(principal * 0.002, 2)
    note_stamps = round(principal * 0.0035, 2)
    rate = round(random.uniform(5.25, 7.75), 3)
    lines = [
        f"Prepared by: {lender}", f"Parcel ID: {parcel()}",
        f"Recorded {date}  OR Book {book}, Page {page}  Instrument #{instr}",
        f"Intangible Tax: ${intangible:,.2f}   Documentary Stamps on Note: ${note_stamps:,.2f}",
        "", "## Mortgage",
        f"THIS MORTGAGE is given on {date}. The mortgagor is {borrower} ('Borrower'). "
        f"This Security Instrument is given to {lender} ('Lender').",
        f"Borrower owes Lender the principal sum of ${principal:,}.00, evidenced by "
        f"Borrower's note dated the same date, with interest at {rate}% per annum, "
        f"with the full debt, if not paid earlier, due and payable on "
        f"{int(date[:4])+30}-{date[5:]}.",
        "This Security Instrument secures to Lender the repayment of the debt and "
        "the performance of Borrower's covenants. Borrower mortgages, grants and "
        "conveys to Lender the following described property in Orange County, Florida:",
        legal_desc(),
        f"which has the address of {prop_addr()} ('Property Address').",
        "Mortgage Electronic Registration Systems, Inc. (MERS) is a nominee for "
        "Lender and Lender's successors and assigns.",
        f"Borrower: {borrower}   Notary: {person()}, Notary Public, State of Florida.",
    ]
    fn = f"Mortgage_ORBook-{book}_Page-{page}_INSTR-{instr}.pdf"
    add("mortgages", fn, "Mortgage", date, lambda p: pdf_doc(p, "Mortgage", lines))


def gen_closing_disclosure(i):
    _, _, _, date = rec_info(i)
    borrower = person(); seller = person(); lender = random.choice(LENDERS)
    loan = random.randint(150, 700) * 1000
    rate = round(random.uniform(5.25, 7.75), 3)
    price = loan + random.randint(20, 120) * 1000
    lines = [
        "## Closing Disclosure",
        f"Closing Date: {date}   Disbursement Date: {date}",
        f"Settlement Agent: {random.choice(TITLE_COS)}   File #: CD-{2025000+i}",
        f"Property: {prop_addr()}   Sale Price: ${price:,}", "",
        f"Borrower: {borrower}     Seller: {seller}     Lender: {lender}", "",
        "## Loan Terms",
        f"Loan Amount: ${loan:,}    Interest Rate: {rate}%    Loan Type: Conventional",
        f"Monthly Principal & Interest: ${int(loan*0.006):,}", "",
        "## Costs at Closing",
        f"Total Closing Costs: ${random.randint(9,18)*1000:,}",
        f"Cash to Close: ${random.randint(20,90)*1000:,}",
        "Loan Costs include Origination Charges, Services You Cannot Shop For, and "
        "Services You Can Shop For. Other Costs include Taxes and Government Fees "
        "(Recording fees, Documentary stamps), Prepaids, and Initial Escrow.",
    ]
    fn = f"Closing_Disclosure_CD-{2025000+i}_{random.choice(CITIES).replace(' ','-')}.pdf"
    add("closing-disclosures", fn, "Closing disclosure", date,
        lambda p: pdf_doc(p, "Closing Disclosure (TRID H-25)", lines))


def gen_title_commitment(i):
    _, _, _, date = rec_info(i)
    tc = random.choice(TITLE_COS); owner = person(); insured = person()
    amount = random.randint(180, 800) * 1000
    lines = [
        f"## ALTA Commitment for Title Insurance", f"Issued by {tc}",
        f"Commitment Number: TC-2025-{800+i:05d}   Effective Date: {date}", "",
        "## Schedule A",
        f"1. Commitment Date: {date}",
        f"2. Policy to be issued: Owner's Policy, Proposed Insured: {insured}, "
        f"Proposed Amount of Insurance: ${amount:,}.",
        f"3. The estate or interest in the Land is Fee Simple, vested in {owner}.",
        "4. The Land is described as follows:", legal_desc(), "",
        "## Schedule B-I (Requirements)",
        "Pay the agreed amounts for the interest to be insured. Pay premiums, fees, "
        "and charges. Record the Deed and the Mortgage to be insured.",
        "## Schedule B-II (Exceptions)",
        "Taxes for the year 2025 and subsequent years. Covenants, conditions and "
        "restrictions of record. Easements as shown on the recorded plat.",
    ]
    fn = f"Title_Commitment_TC-2025-{800+i:05d}.pdf"
    add("title-commitments", fn, "Title commitment", date,
        lambda p: pdf_doc(p, "ALTA Title Commitment", lines))


def gen_condo_disclosure(i):
    date = rec_info(i)[3]; assoc = f"{random.choice(SUBS)} Condominium Association, Inc."
    lines = [
        "## Governance Form / Condominium Disclosure (Fla. Stat. 718.503)",
        f"Association: {assoc}", f"Prepared: {date}",
        "IF YOU HAVE ANY QUESTIONS REGARDING YOUR RIGHTS AND RESPONSIBILITIES AS AN "
        "OWNER, CONTACT THE ASSOCIATION OR THE DIVISION OF FLORIDA CONDOMINIUMS.",
        f"1. As an owner you are responsible for a monthly assessment of "
        f"${random.randint(280,650)}, subject to change.",
        "2. You may be obligated to pay special assessments to the association.",
        "3. The association may have the power to place a lien on your unit for "
        "unpaid assessments.",
        "4. There are restrictions on leasing your unit; see the Declaration.",
        "5. This document is a summary; the Declaration of Condominium, Bylaws, and "
        "Articles of Incorporation govern.",
    ]
    fn = f"Condo_Disclosure_{assoc.split(' Condominium')[0].replace(' ','-')}_{i:02d}.pdf"
    add("condo-hoa-disclosures", fn, "Condo disclosure", date,
        lambda p: pdf_doc(p, "Condominium Governance Disclosure", lines))


def gen_hoa_disclosure(i):
    date = rec_info(i)[3]; assoc = f"{random.choice(SUBS)} Homeowners Association, Inc."
    paras = [
        "DISCLOSURE SUMMARY FOR " + assoc.upper() + " (Fla. Stat. 720.401)",
        f"Prepared: {date}",
        "1. AS A PURCHASER OF PROPERTY IN THIS COMMUNITY, YOU WILL BE OBLIGATED TO "
        "BE A MEMBER OF A HOMEOWNERS' ASSOCIATION.",
        f"2. THERE HAVE BEEN OR WILL BE RECORDED RESTRICTIVE COVENANTS GOVERNING THE "
        f"USE AND OCCUPANCY OF PROPERTIES IN THIS COMMUNITY.",
        f"3. YOU WILL BE OBLIGATED TO PAY ASSESSMENTS OF APPROXIMATELY "
        f"${random.randint(150,500)} PER QUARTER, SUBJECT TO PERIODIC CHANGE.",
        "4. YOUR FAILURE TO PAY ASSESSMENTS COULD RESULT IN A LIEN ON YOUR PROPERTY.",
        "5. THE DEVELOPER MAY HAVE THE RIGHT TO AMEND THE RESTRICTIVE COVENANTS.",
        "THESE DOCUMENTS ARE EITHER MATTERS OF PUBLIC RECORD OR WILL BE PROVIDED TO YOU.",
    ]
    fn = f"HOA_Disclosure_Summary_{assoc.split(' Homeowners')[0].replace(' ','-')}_{i:02d}.docx"
    def w(p):
        d = Document(); d.add_heading("HOA Disclosure Summary", level=1)
        for para in paras: d.add_paragraph(para)
        d.save(str(p))
    add("condo-hoa-disclosures", fn, "HOA disclosure", date, w)


def gen_filled_contract(i):
    date = rec_info(i)[3]; buyer = person(); seller = person()
    price = random.randint(200, 750) * 1000
    paras = [
        f"'AS IS' Residential Contract For Sale And Purchase (sample, filled) - {date}",
        f"SELLER: {seller}", f"BUYER: {buyer}",
        f"PROPERTY: {prop_addr()}   County: Orange",
        f"Legal Description: {legal_desc()}",
        f"PURCHASE PRICE: ${price:,}.00",
        f"Initial deposit: ${int(price*0.02):,}   Additional deposit: ${int(price*0.03):,}",
        f"Financing: Conventional loan of ${int(price*0.8):,}. Closing on or before "
        f"{int(date[:4])}-{(int(date[5:7])%12)+1:02d}-15.",
        "This is a FloridaRealtors/FloridaBar 'AS IS' sample completed with fictitious "
        "parties for demonstration only.",
    ]
    fn = f"AS-IS_Contract_FILLED_{buyer.replace(' ','-')}_{i:03d}.docx"
    def w(p):
        d = Document(); d.add_heading("AS-IS Residential Contract (filled sample)", level=1)
        for para in paras: d.add_paragraph(para)
        d.save(str(p))
    add("filled-contracts", fn, "AS-IS contract (filled sample)", date, w)


def build_synthetic():
    for i in range(1, 91): gen_deed(i)                 # 90 deeds
    for i in range(1, 81): gen_mortgage(i)             # 80 mortgages
    for i in range(1, 41): gen_closing_disclosure(i)   # 40 closing disclosures
    for i in range(1, 31): gen_title_commitment(i)     # 30 title commitments
    for i in range(1, 21): gen_condo_disclosure(i)     # 20 condo disclosures
    for i in range(1, 21): gen_hoa_disclosure(i)       # 20 HOA disclosures
    for i in range(1, 13): gen_filled_contract(i)      # 12 filled contracts


def write_manifest():
    real = sum(1 for r in rows if r[4] == "real-download")
    syn = sum(1 for r in rows if r[4] == "synthetic")
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Florida closing-document demo corpus. total={len(rows)}; "
                f"real_downloaded={real}; synthetic={syn}; cap=400. "
                f"Synthetic files stand in for Orange County recorded deeds/mortgages "
                f"(county portal has no API/bulk download).\n")
        w = csv.writer(f)
        w.writerow(["filename", "type", "source", "date", "origin", "source_url"])
        w.writerows(sorted(rows))
    return real, syn


if __name__ == "__main__":
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    print("=== downloading real public PDFs ===")
    n_real = download_real()
    print("=== generating synthetic FL closing docs ===")
    build_synthetic()
    real, syn = write_manifest()
    total = len(rows)
    print("----")
    print(f"Corpus total: {total}  (real={real}, synthetic={syn})  -> {CORPUS}")
    print(f"Manifest: {MANIFEST}")
    if not (150 <= total <= 400):
        print(f"WARNING: total {total} outside 150-400 target")
