"""Invoice processing module for extracting and validating invoice data."""

import re
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.extraction_result import ExtractionResult


class ValidationStatus(str, Enum):
    """Invoice validation status."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNCHECKED = "unchecked"


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(..., description="Item description")
    quantity: float = Field(default=1.0, description="Quantity")
    unit_price: float = Field(default=0.0, description="Unit price")
    total: float = Field(default=0.0, description="Line item total")


class InvoiceData(BaseModel):
    """Complete extracted invoice data."""

    vendor: Optional[str] = Field(None, description="Vendor/supplier name")
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    date: Optional[str] = Field(None, description="Invoice date")
    due_date: Optional[str] = Field(None, description="Payment due date")
    line_items: List[LineItem] = Field(
        default_factory=list, description="Line items on the invoice"
    )
    subtotal: float = Field(default=0.0, description="Subtotal before tax")
    tax: float = Field(default=0.0, description="Tax amount")
    total: float = Field(default=0.0, description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")
    po_number: Optional[str] = Field(None, description="Purchase order number")
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.UNCHECKED, description="Validation result"
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation error messages"
    )


class InvoiceProcessor:
    """Processes invoice documents to extract and validate data.

    Extracts line items, validates totals, and checks for common issues.
    """

    def process(self, extraction_result: ExtractionResult) -> InvoiceData:
        """Process an extraction result as an invoice.

        Args:
            extraction_result: The extraction result containing invoice text.

        Returns:
            InvoiceData with extracted and validated invoice information.
        """
        text = extraction_result.raw_text or ""

        # Extract basic invoice fields
        vendor = self._extract_vendor(text)
        invoice_number = self._extract_invoice_number(text)
        date = self._extract_date(text, "date")
        due_date = self._extract_date(text, "due_date")
        po_number = self._extract_po_number(text)
        currency = self._extract_currency(text)

        # Extract line items
        line_items = self.extract_line_items(text)

        # Extract totals
        subtotal = self._extract_amount(text, "subtotal")
        tax = self._extract_amount(text, "tax")
        total = self._extract_amount(text, "total")

        # If no total found, compute from line items
        if total == 0.0 and line_items:
            total = sum(item.total for item in line_items)

        if subtotal == 0.0 and line_items:
            subtotal = sum(item.total for item in line_items)

        invoice = InvoiceData(
            vendor=vendor,
            invoice_number=invoice_number,
            date=date,
            due_date=due_date,
            line_items=line_items,
            subtotal=subtotal,
            tax=tax,
            total=total,
            currency=currency,
            po_number=po_number,
        )

        # Validate amounts
        validation_errors = self.validate_amounts(line_items, total)
        if validation_errors:
            invoice.validation_status = ValidationStatus.WARNING
            invoice.validation_errors = validation_errors
        elif line_items:
            invoice.validation_status = ValidationStatus.VALID

        return invoice

    def extract_line_items(self, text: str) -> List[LineItem]:
        """Parse line items from invoice text.

        Args:
            text: Invoice text to parse.

        Returns:
            List of extracted line items.
        """
        line_items: List[LineItem] = []
        if not text:
            return line_items

        # Pattern: description, quantity, unit price, total
        # Matches lines like: "Widget A    2    $10.00    $20.00"
        patterns = [
            # qty x price = total pattern
            r"(.+?)\s+(\d+(?:\.\d+)?)\s+\$?([\d,]+\.\d{2})\s+\$?([\d,]+\.\d{2})",
            # Description followed by amount
            r"(.+?)\s{2,}(\d+)\s+@\s+\$?([\d,]+\.\d{2})\s+=?\s*\$?([\d,]+\.\d{2})",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                desc = match.group(1).strip()
                qty = float(match.group(2))
                unit_price = float(match.group(3).replace(",", ""))
                total = float(match.group(4).replace(",", ""))

                # Skip if description looks like a header
                if desc.lower() in ("description", "item", "product", "service"):
                    continue

                line_items.append(
                    LineItem(
                        description=desc,
                        quantity=qty,
                        unit_price=unit_price,
                        total=total,
                    )
                )

            if line_items:
                break

        # Simpler pattern: just description and amount
        if not line_items:
            simple_pattern = r"[-*]\s*(.+?)\s+\$?([\d,]+\.\d{2})"
            matches = re.finditer(simple_pattern, text)
            for match in matches:
                desc = match.group(1).strip()
                total = float(match.group(2).replace(",", ""))
                line_items.append(
                    LineItem(
                        description=desc,
                        quantity=1.0,
                        unit_price=total,
                        total=total,
                    )
                )

        return line_items

    def validate_amounts(
        self, line_items: List[LineItem], total: float
    ) -> List[str]:
        """Cross-check line item sum against stated total.

        Args:
            line_items: Extracted line items.
            total: Stated total amount.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if not line_items or total == 0.0:
            return errors

        computed_total = sum(item.total for item in line_items)

        # Allow for small floating point differences and tax
        if abs(computed_total - total) > 0.01:
            diff = total - computed_total
            if diff > 0:
                errors.append(
                    f"Line items sum ({computed_total:.2f}) is less than "
                    f"stated total ({total:.2f}). Difference: {diff:.2f} "
                    f"(may be tax or shipping)"
                )
            else:
                errors.append(
                    f"Line items sum ({computed_total:.2f}) exceeds "
                    f"stated total ({total:.2f}). Difference: {abs(diff):.2f}"
                )

        # Check individual line items
        for item in line_items:
            expected = item.quantity * item.unit_price
            if abs(expected - item.total) > 0.01:
                errors.append(
                    f"Line item '{item.description}': "
                    f"qty({item.quantity}) x price({item.unit_price}) = "
                    f"{expected:.2f}, but stated as {item.total:.2f}"
                )

        return errors

    def detect_duplicates(
        self, invoice_data: InvoiceData, history: List[InvoiceData]
    ) -> List[str]:
        """Check for duplicate invoices in history.

        Args:
            invoice_data: Current invoice to check.
            history: List of previously processed invoices.

        Returns:
            List of potential duplicate warnings.
        """
        warnings: List[str] = []

        for prev in history:
            if (
                invoice_data.invoice_number
                and prev.invoice_number
                and invoice_data.invoice_number == prev.invoice_number
            ):
                warnings.append(
                    f"Duplicate invoice number: {invoice_data.invoice_number}"
                )

            if (
                invoice_data.vendor == prev.vendor
                and invoice_data.total == prev.total
                and invoice_data.date == prev.date
            ):
                warnings.append(
                    f"Possible duplicate: same vendor ({invoice_data.vendor}), "
                    f"amount ({invoice_data.total}), and date ({invoice_data.date})"
                )

        return warnings

    def match_vendor(
        self, vendor_name: str, vendor_db: List[str]
    ) -> Optional[str]:
        """Fuzzy match vendor name against known vendors.

        Args:
            vendor_name: Vendor name from the invoice.
            vendor_db: List of known vendor names.

        Returns:
            Best matching vendor name or None if no good match.
        """
        if not vendor_name or not vendor_db:
            return None

        vendor_lower = vendor_name.lower().strip()

        # Exact match first
        for known in vendor_db:
            if known.lower().strip() == vendor_lower:
                return known

        # Partial match
        for known in vendor_db:
            known_lower = known.lower().strip()
            if vendor_lower in known_lower or known_lower in vendor_lower:
                return known

        # Token-based similarity
        vendor_tokens = set(vendor_lower.split())
        best_match = None
        best_score = 0.0

        for known in vendor_db:
            known_tokens = set(known.lower().strip().split())
            if not known_tokens:
                continue
            intersection = vendor_tokens & known_tokens
            union = vendor_tokens | known_tokens
            score = len(intersection) / len(union) if union else 0.0
            if score > best_score and score > 0.5:
                best_score = score
                best_match = known

        return best_match

    def _extract_vendor(self, text: str) -> Optional[str]:
        """Extract vendor name from invoice text."""
        patterns = [
            r"(?:From|Vendor|Supplier|Bill\s+From)[:\s]+([^\n]+)",
            r"^([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Ltd|Co)\.?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number from text."""
        patterns = [
            r"(?:Invoice|Inv)\s*(?:#|No\.?|Number)[:\s]*([A-Z0-9-]+)",
            r"(?:Invoice|Inv)[:\s]+([A-Z0-9-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_date(self, text: str, date_type: str) -> Optional[str]:
        """Extract date from text based on type."""
        if date_type == "due_date":
            patterns = [
                r"(?:Due\s+Date|Payment\s+Due)[:\s]+([^\n,]+)",
                r"(?:Due)[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})",
            ]
        else:
            patterns = [
                r"(?:Invoice\s+Date|Date)[:\s]+([^\n,]+)",
                r"(?:Date)[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})",
            ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_po_number(self, text: str) -> Optional[str]:
        """Extract purchase order number."""
        patterns = [
            r"(?:PO|P\.O\.|Purchase\s+Order)\s*(?:#|No\.?|Number)?[:\s]*([A-Z0-9-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_currency(self, text: str) -> str:
        """Detect currency from text."""
        if re.search(r"[£]|GBP", text):
            return "GBP"
        if re.search(r"[€]|EUR", text):
            return "EUR"
        return "USD"

    def _extract_amount(self, text: str, amount_type: str) -> float:
        """Extract a specific amount from invoice text."""
        if amount_type == "subtotal":
            patterns = [r"(?:Sub\s*total|Subtotal)[:\s]*\$?([\d,]+\.\d{2})"]
        elif amount_type == "tax":
            patterns = [r"(?:Tax|VAT|GST|HST)[:\s]*\$?([\d,]+\.\d{2})"]
        else:
            patterns = [
                r"(?:Total\s+Due|Amount\s+Due|Grand\s+Total|Total)[:\s]*\$?([\d,]+\.\d{2})",
                r"(?:Balance\s+Due)[:\s]*\$?([\d,]+\.\d{2})",
            ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", ""))
        return 0.0
