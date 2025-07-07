import re
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import openai
import os

# --- Pydantic Models for Structured Output ---
# These models define the schema for the data you want to extract.
# You will create a new model for each type of insurance policy.

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor_name: Optional[str] = None
    customer_name: Optional[str] = None
    total_amount: Optional[str] = None
    currency: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = []

class ReceiptData(BaseModel):
    store_name: Optional[str] = None
    transaction_date: Optional[str] = None
    total_amount: Optional[str] = None
    currency: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = []

class ContractSummaryData(BaseModel):
    contract_title: Optional[str] = None
    parties: Optional[List[str]] = []
    effective_date: Optional[str] = None
    termination_clause_summary: Optional[str] = None
    governing_law: Optional[str] = None

class WorkersCompPolicyData(BaseModel):
    """
    Pydantic model for Workers Compensation Policy data points,
    based on "All LOB - Data Points.pdf" and sample policies.
    """
    name_insured: Optional[str] = Field(None, description="The primary name insured on the policy.")
    other_named_insured: Optional[List[str]] = Field(None, description="Other named insureds on the policy.")
    mailing_address: Optional[str] = Field(None, description="The mailing address of the insured.")
    policy_number: Optional[str] = Field(None, description="The unique policy number.")
    policy_period_start: Optional[str] = Field(None, description="The start date of the policy period (e.g., MM/DD/YYYY).")
    policy_period_end: Optional[str] = Field(None, description="The end date of the policy period (e.g., MM/DD/YYYY).")
    issuing_company: Optional[str] = Field(None, description="The name of the insurance company issuing the policy.")
    premium: Optional[str] = Field(None, description="The total premium for the policy.")
    paid_in_full_discount: Optional[str] = Field(None, description="Any paid in full discount amount.")
    miscellaneous_premium: Optional[str] = Field(None, description="Any miscellaneous premium amount.")
    location: Optional[str] = Field(None, description="The location covered by the policy.")
    general_liability_limits: Optional[Dict[str, str]] = Field(None, description="General Liability limits if applicable (e.g., 'Each Occurrence', 'General Aggregate').")
    employers_liability_limits: Optional[Dict[str, str]] = Field(None, description="Employers Liability limits (e.g., 'Each Accident', 'Disease - Each Employee', 'Disease - Policy Limit').")
    deductible: Optional[str] = Field(None, description="The deductible amount.")
    terrorism_coverage: Optional[str] = Field(None, description="Indication if terrorism coverage is included or excluded.")
    exclusions_summary: Optional[str] = Field(None, description="A summary of any significant exclusions.")
    additional_interest: Optional[List[str]] = Field(None, description="Any additional interests listed.")
    forms_and_endorsements: Optional[List[str]] = Field(None, description="List of forms and endorsements attached to the policy.")
    business_classification: Optional[str] = Field(None, description="The business classification or nature of operations.")
    retro_date: Optional[str] = Field(None, description="The retro date for claims-made policies.")
    prior_and_pending_date: Optional[str] = Field(None, description="The prior and pending date for claims-made policies.")
    continuity_date: Optional[str] = Field(None, description="The continuity date for claims-made policies.")
    underlying_insurance: Optional[str] = Field(None, description="Details of any underlying insurance.")
    
class DataExtractor:
    def __init__(self):
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            print("Warning: OPENAI_API_KEY environment variable not set. LLM-based extraction will not work.")

    def extract_data(self, text: str, doc_type: str) -> Optional[Dict[str, Any]]:
        """
        Extracts specific data based on the document type.
        This function routes to different extraction methods.
        """
        if doc_type == "invoice":
            return self._extract_invoice_data(text)
        elif doc_type == "receipt":
            return self._extract_receipt_data(text)
        elif doc_type == "contract_summary":
            return self._extract_contract_summary(text)
        elif doc_type == "workers_comp":
            return self._extract_workers_comp_data(text)
        else:
            print(f"Warning: No specific extraction logic for document type '{doc_type}'. Attempting generic LLM extraction.")
            return self._generic_llm_extraction(text, doc_type)

    def _extract_invoice_data(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts invoice-specific data.
        Combines rule-based (regex) with LLM for robustness.
        """
        extracted = {}

        # --- Rule-based Extraction (Regex) for common fields ---
        # Invoice Number
        invoice_number_match = re.search(r'(invoice|inv|bill)\s*#?[:\s]*([A-Za-z0-9-]+)', text, re.IGNORECASE)
        if invoice_number_match:
            extracted['invoice_number'] = invoice_number_match.group(2).strip()

        # Dates (can be complex, simplified for example)
        date_match = re.search(r'(date|invoice date|bill date)[:\s]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', text, re.IGNORECASE)
        if date_match:
            extracted['invoice_date'] = date_match.group(2).strip()
        
        due_date_match = re.search(r'(due date)[:\s]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', text, re.IGNORECASE)
        if due_date_match:
            extracted['due_date'] = due_date_match.group(2).strip()

        # Total Amount (basic regex, needs improvement for currency codes etc.)
        total_amount_match = re.search(r'(total|amount due|balance due)[:\s]*[A-Z$€£]*\s*([\d.,]+)', text, re.IGNORECASE)
        if total_amount_match:
            extracted['total_amount'] = total_amount_match.group(2).replace(',', '').strip()
            currency_match = re.search(r'(total|amount due|balance due)[:\s]*([A-Z$€£]+)\s*[\d.,]+', text, re.IGNORECASE)
            if currency_match:
                extracted['currency'] = currency_match.group(2).strip()
            else:
                extracted['currency'] = '$' # Default if not found

        # --- LLM-based extraction for more nuanced or missing fields ---
        if self.openai_client:
            try:
                # Define a Pydantic model for the desired output
                # Using the `response_model` parameter is a conceptual representation for clarity.
                # In actual OpenAI API calls, you typically receive a string and parse it.
                prompt = (
                    "You are an AI assistant specialized in extracting structured data from invoices. "
                    "Extract the following fields from the invoice text: invoice_number, invoice_date, "
                    "due_date, vendor_name, customer_name, total_amount, currency, and line items (description and amount). "
                    "Return the data as a JSON object, adhering strictly to the InvoiceData Pydantic model schema. "
                    "If a field is not found, omit it or set its value to null. For items, provide a list of objects with 'description' and 'amount'."
                    f"\n\nInvoice Text:\n{text}\n\nInvoiceData Schema:\n{InvoiceData.model_json_schema()}"
                )
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo", # Or gpt-4, gpt-4o for better results
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specialized in extracting structured data from invoices. Return the data as a JSON object."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                llm_output_str = response.choices[0].message.content
                llm_extracted_model = self._parse_llm_json_output(llm_output_str, InvoiceData)

                if llm_extracted_model:
                    # Merge rule-based and LLM-based results, LLM can fill gaps
                    # LLM results take precedence for fields it found, but we keep rule-based if LLM missed it
                    extracted.update(llm_extracted_model.dict(exclude_unset=True)) 

                return InvoiceData(**extracted).dict(exclude_none=True) # Return validated and cleaned data
            except Exception as e:
                print(f"Error during LLM invoice extraction: {e}")
                # Fallback to only rule-based if LLM fails
                return InvoiceData(**extracted).dict(exclude_none=True)
        
        return InvoiceData(**extracted).dict(exclude_none=True)

    def _extract_receipt_data(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts receipt-specific data.
        """
        extracted = {}

        # Rule-based (example)
        store_match = re.search(r'(store|shop|merchant)\s*name[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        if store_match:
            extracted['store_name'] = store_match.group(2).strip()

        # LLM-based extraction
        if self.openai_client:
            try:
                prompt = (
                    "You are an AI assistant specialized in extracting structured data from receipts. "
                    "Extract the following fields from the receipt text: store_name, transaction_date, "
                    "total_amount, currency, and line items (description and amount). "
                    "Return the data as a JSON object, adhering strictly to the ReceiptData Pydantic model schema. "
                    "If a field is not found, omit it or set its value to null. For items, provide a list of objects with 'description' and 'amount'."
                    f"\n\nReceipt Text:\n{text}\n\nReceiptData Schema:\n{ReceiptData.model_json_schema()}"
                )
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specialized in extracting structured data from receipts. Return the data as a JSON object."},
                        {"role": "user", "content": prompt}
                    ]
                )
                llm_output_str = response.choices[0].message.content
                llm_extracted_model = self._parse_llm_json_output(llm_output_str, ReceiptData)
                if llm_extracted_model:
                    extracted.update(llm_extracted_model.dict(exclude_unset=True))
                return ReceiptData(**extracted).dict(exclude_none=True)
            except Exception as e:
                print(f"Error during LLM receipt extraction: {e}")
                return ReceiptData(**extracted).dict(exclude_none=True)
        
        return ReceiptData(**extracted).dict(exclude_none=True)

    def _extract_contract_summary(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts key information for a contract summary.
        This is typically heavily LLM-dependent due to unstructured nature.
        """
        extracted = {}
        if self.openai_client:
            try:
                prompt = (
                    "You are an AI assistant specialized in summarizing key aspects of legal contracts. "
                    "Extract the contract_title, parties, effective_date, termination_clause_summary, and governing_law. "
                    "Return as a JSON object, adhering strictly to the ContractSummaryData Pydantic model schema. "
                    "If a field is not found, omit it or set its value to null. "
                    "For parties, provide a list of names."
                    f"\n\nContract Text:\n{text}\n\nContractSummaryData Schema:\n{ContractSummaryData.model_json_schema()}"
                )
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specialized in summarizing key aspects of legal contracts. Return the data as a JSON object."},
                        {"role": "user", "content": prompt}
                    ]
                )
                llm_output_str = response.choices[0].message.content
                llm_extracted_model = self._parse_llm_json_output(llm_output_str, ContractSummaryData)
                if llm_extracted_model:
                    extracted.update(llm_extracted_model.dict(exclude_unset=True))
                return ContractSummaryData(**extracted).dict(exclude_none=True)
            except Exception as e:
                print(f"Error during LLM contract summary extraction: {e}")
                return None # No fallback for contract summary if LLM fails
        print("OpenAI API key not found. Cannot perform LLM-based contract summary extraction.")
        return None

    def _extract_workers_comp_data(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts Workers Compensation Policy data points using LLM.
        """
        extracted = {}
        if self.openai_client:
            try:
                prompt = (
                    "You are an AI assistant specialized in extracting structured data from Workers Compensation Insurance Policies. "
                    "Extract the following fields from the policy text: "
                    "name_insured, other_named_insured (as a list), mailing_address, policy_number, "
                    "policy_period_start (format MM/DD/YYYY), policy_period_end (format MM/DD/YYYY), "
                    "issuing_company, premium, paid_in_full_discount, miscellaneous_premium, location, "
                    "general_liability_limits (as a dictionary of limit types and values), "
                    "employers_liability_limits (as a dictionary of limit types and values), "
                    "deductible, terrorism_coverage (e.g., 'Included', 'Excluded'), "
                    "exclusions_summary (a brief summary), additional_interest (as a list), "
                    "forms_and_endorsements (as a list), business_classification, retro_date, "
                    "prior_and_pending_date, and continuity_date, underlying_insurance. "
                    "Return the data as a JSON object, adhering strictly to the WorkersCompPolicyData Pydantic model schema. "
                    "If a field is not found, omit it or set its value to null. "
                    "For dates, ensure MM/DD/YYYY format if possible."
                    f"\n\nWorkers Compensation Policy Text:\n{text}\n\nWorkersCompPolicyData Schema:\n{WorkersCompPolicyData.model_json_schema()}"
                )
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o", # Using a more capable model for better accuracy on complex documents
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specialized in extracting structured data from Workers Compensation Insurance Policies. Return the data as a JSON object."},
                        {"role": "user", "content": prompt}
                    ]
                )
                llm_output_str = response.choices[0].message.content
                llm_extracted_model = self._parse_llm_json_output(llm_output_str, WorkersCompPolicyData)
                if llm_extracted_model:
                    extracted.update(llm_extracted_model.dict(exclude_unset=True))
                return WorkersCompPolicyData(**extracted).dict(exclude_none=True)
            except Exception as e:
                print(f"Error during LLM Workers Comp extraction: {e}")
                return None
        print("OpenAI API key not found. Cannot perform LLM-based Workers Comp extraction.")
        return None

    def _generic_llm_extraction(self, text: str, doc_type: str) -> Optional[Dict[str, Any]]:
        """
        A generic LLM extraction for unspecified document types.
        It asks the LLM to extract "key information" based on the doc_type hint.
        """
        if not self.openai_client:
            print("OpenAI API key not found. Cannot perform generic LLM extraction.")
            return None

        try:
            prompt = (
                f"The following document is identified as a '{doc_type}'. "
                "Extract all key information and relevant entities from this document. "
                "Structure the output as a JSON object with meaningful keys and values. "
                "If there are lists (e.g., line items, multiple parties), represent them as JSON arrays."
                f"\n\nDocument Text:\n{text}"
            )
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant for document parsing. Extract key information from the provided document text and format it as a JSON object."},
                    {"role": "user", "content": prompt}
                ]
            )
            llm_output_str = response.choices[0].message.content
            # Attempt to parse as general JSON
            try:
                return json.loads(llm_output_str)
            except json.JSONDecodeError:
                print(f"Warning: Generic LLM extraction did not return valid JSON. Raw output: {llm_output_str[:200]}...")
                return {"extracted_text": text, "llm_raw_output": llm_output_str} # Return raw output for debugging
        except Exception as e:
            print(f"An error occurred during generic LLM extraction: {e}")
            return {"extracted_text": text, "error": str(e)}

    def _parse_llm_json_output(self, llm_output_str: str, model: BaseModel) -> Optional[BaseModel]:
        """Parses LLM string output into a Pydantic model, handling common issues."""
        try:
            # LLMs sometimes wrap JSON in markdown code blocks
            if llm_output_str.strip().startswith("```json"):
                llm_output_str = llm_output_str.strip()[7:]
                if llm_output_str.endswith("```"):
                    llm_output_str = llm_output_str[:-3]
            
            data = json.loads(llm_output_str)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM output as JSON or validate with Pydantic: {e}")
            print(f"LLM Raw Output: {llm_output_str}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during LLM output parsing: {e}")
            return None
