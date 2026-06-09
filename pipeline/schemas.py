from typing import List
from pydantic import BaseModel, Field


class Product(BaseModel):
    """Un produs sau serviciu de pe factură."""

    denumire:  str   = Field(description="Denumirea produsului sau serviciului")
    cantitate: int   = Field(description="Cantitatea facturată")
    pret:      float = Field(description="Prețul unitar în RON")
    total:     float = Field(description="Valoarea totală pe rând (cantitate × preț unitar) în RON")


class RomanianInvoice(BaseModel):
    """Schemă pentru extracție date factură românească."""

    numar:    str   = Field(description="Numărul facturii (ex: FV-2024-001)")
    data:     str   = Field(description="Data emiterii (format: DD.MM.YYYY)")
    client:   str   = Field(description="Numele clientului (compania facturată)")
    furnizor: str   = Field(description="Numele furnizorului (compania care emite factura)")
    total:    float = Field(description="Suma totală de plată în RON (cu TVA inclus)")

    produse: List[Product] = Field(
        default=[],
        description="Lista produselor/serviciilor facturate",
    )
