"""Tests for the future-option (options-on-futures) resolver."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from security_universe import Security, SecurityType
from security_universe.models import OptionType, SettlementType
from security_universe.resolvers import (
    FutureOptionSecurityIdResolver,
    ResolverChain,
    parse_future_option_symbol,
)

ES_CALL = "./ESU6 E1CN6 260701C6325"


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def test_parses_all_three_parts() -> None:
    p = parse_future_option_symbol(ES_CALL)
    assert p is not None
    assert p.underlying_future == "ESU6"
    assert p.product_code == "E1CN6"  # the token canonical.py drops
    assert p.expiry == date(2026, 7, 1)
    assert p.option_type == OptionType.CALL
    assert p.strike == Decimal("6325")


def test_parses_put() -> None:
    p = parse_future_option_symbol("./ESU6 E1CN6 260701P6300")
    assert p is not None and p.option_type == OptionType.PUT
    assert p.strike == Decimal("6300")


def test_parses_decimal_strike() -> None:
    p = parse_future_option_symbol("./CLU6 LO1U6 260815C72.5")
    assert p is not None and p.strike == Decimal("72.5")


def test_whitespace_and_lowercase_normalized() -> None:
    p = parse_future_option_symbol("  ./esu6 e1cn6 260701c6325  ")
    assert p is not None and p.underlying_future == "ESU6"
    assert p.product_code == "E1CN6"


def test_quarterly_product_equals_future_code() -> None:
    p = parse_future_option_symbol("./ESU6 ESU6 260918C6400")
    assert p is not None and p.product_code == "ESU6"


def test_bare_symbol_without_series_falls_back_to_future_code() -> None:
    p = parse_future_option_symbol("./ESU6 260701C6325")
    assert p is not None and p.product_code == "ESU6"


def test_non_future_option_symbols_do_not_parse() -> None:
    assert parse_future_option_symbol("AAPL  250117C00150000") is None  # OCC
    assert parse_future_option_symbol("/ESU6") is None  # bare future
    assert parse_future_option_symbol("AAPL") is None  # equity


# --------------------------------------------------------------------------- #
# packed head — a 3-char root fills its fixed-width field, leaving no separator
# (TT transaction symbols for micros: MES/MNQ/MCL/BTC/M2K)
# --------------------------------------------------------------------------- #
def test_parses_packed_weekly_series() -> None:
    p = parse_future_option_symbol("./MESU5EXQ5  250829P6430")
    assert p is not None
    assert p.underlying_future == "MESU5"
    assert p.product_code == "EXQ5"
    assert p.expiry == date(2025, 8, 29)
    assert p.option_type == OptionType.PUT
    assert p.strike == Decimal("6430")


def test_parses_packed_quarterly_product_equals_future_code() -> None:
    p = parse_future_option_symbol("./MESH4MESH4 240315C4950")
    assert p is not None
    assert p.underlying_future == "MESH4"
    assert p.product_code == "MESH4"


def test_parses_packed_decimal_strike() -> None:
    p = parse_future_option_symbol("./MCLG6MCOG6 260114C49.5")
    assert p is not None
    assert p.underlying_future == "MCLG6"
    assert p.product_code == "MCOG6"
    assert p.strike == Decimal("49.5")


def test_parses_packed_five_char_series() -> None:
    p = parse_future_option_symbol("./MNQM6D2CM6 260610P29600")
    assert p is not None
    assert p.underlying_future == "MNQM6"
    assert p.product_code == "D2CM6"


def test_packed_head_assumes_series_codes_start_with_a_letter() -> None:
    # Documented limitation: with a 1-digit contract year, a leading-digit series
    # code would be eaten by the greedy year match and mis-split the head. No CME
    # futures-option series code starts with a digit (E1A/EW4/EX/MCO/D2C/LO1/VY1
    # all lead with letters); this pins the assumption so a violation fails loudly.
    p = parse_future_option_symbol("./MESU51XQ5  250829P6430")
    assert p is not None and p.underlying_future != "MESU5"


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
def test_resolves_es_call_with_enrichment() -> None:
    s = FutureOptionSecurityIdResolver.default().resolve(
        Security(symbol=ES_CALL, security_type=SecurityType.FUTURE_OPTION)
    )
    assert s.security_type == SecurityType.FUTURE_OPTION
    assert s.security_id == "future_option:ES:U6:2026-07-01:call:6325"
    assert s.short_name == "ESU6 6325C 2026-07-01"
    assert s.root_symbol == "ES"
    assert s.underlying == "ESU6"
    assert s.option_root == "E1CN6"
    assert s.expiry == date(2026, 7, 1)
    assert s.strike == Decimal("6325")
    assert s.option_type == OptionType.CALL
    assert s.settlement_type == SettlementType.PHYSICAL
    assert s.contract_multiplier == Decimal("50")


def test_order_symbol_carries_native_symbol_for_round_trip() -> None:
    s = FutureOptionSecurityIdResolver.default().resolve(Security(symbol=ES_CALL))
    # The opaque order symbol round-trips off the Security itself (no registry).
    assert s.order_symbol == ES_CALL
    # Its decomposed parts live in the natural fields, not metadata.
    assert s.option_root == "E1CN6"          # the CME series/product code
    assert s.underlying == "ESU6"            # the underlying future contract
    assert s.metadata == {}                  # nothing stuffed into metadata


def test_self_detects_without_security_type() -> None:
    s = FutureOptionSecurityIdResolver.default().resolve(Security(symbol=ES_CALL))
    assert s.security_type == SecurityType.FUTURE_OPTION
    assert s.security_id == "future_option:ES:U6:2026-07-01:call:6325"


def test_resolves_packed_micro_and_round_trips_order_symbol() -> None:
    packed = "./MESU5EXQ5  250829P6430"
    s = FutureOptionSecurityIdResolver.default().resolve(Security(symbol=packed))
    assert s.security_type == SecurityType.FUTURE_OPTION
    assert s.security_id == "future_option:MES:U5:2025-08-29:put:6430"
    assert s.root_symbol == "MES"
    assert s.underlying == "MESU5"
    assert s.option_root == "EXQ5"
    assert s.contract_multiplier == Decimal("5")
    # to_tt returns the carried opaque order symbol byte-identical — packed spacing intact.
    from security_universe.symbology import to_tt

    assert to_tt(s) == packed


def test_micros_and_non_equity_products_enrich_from_rules() -> None:
    res = FutureOptionSecurityIdResolver.default()
    mes = res.resolve(Security(symbol="./MESU6 E1CN6 260701C6325"))
    assert mes.root_symbol == "MES" and mes.contract_multiplier == Decimal("5")
    cl = res.resolve(Security(symbol="./CLU6 LO1U6 260815C72"))
    assert cl.root_symbol == "CL" and cl.contract_multiplier == Decimal("1000")


def test_unknown_root_falls_back_to_physical_no_multiplier() -> None:
    s = FutureOptionSecurityIdResolver.default().resolve(
        Security(symbol="./ZZU6 ZZ1U6 260701C100")
    )
    assert s.root_symbol == "ZZ"
    assert s.settlement_type == SettlementType.PHYSICAL
    assert s.contract_multiplier is None


def test_non_future_option_passes_through_unchanged() -> None:
    original = Security(symbol="AAPL", security_type=SecurityType.STOCK)
    out = FutureOptionSecurityIdResolver.default().resolve(original)
    assert out == original
    assert out.security_id is None


def test_explicit_fields_are_preserved() -> None:
    s = FutureOptionSecurityIdResolver.default().resolve(
        Security(
            symbol=ES_CALL,
            security_id="custom:id",
            short_name="my name",
            contract_multiplier=Decimal("99"),
        )
    )
    assert s.security_id == "custom:id"
    assert s.short_name == "my name"
    assert s.contract_multiplier == Decimal("99")


def test_original_security_not_mutated() -> None:
    original = Security(symbol=ES_CALL)
    FutureOptionSecurityIdResolver.default().resolve(original)
    assert original.security_id is None
    assert original.metadata == {}


def test_from_yaml_overrides_rules(tmp_path) -> None:
    rule_file = tmp_path / "rules.yaml"
    rule_file.write_text("ES: { settlement_type: cash, contract_multiplier: 1 }\n")
    res = FutureOptionSecurityIdResolver.from_yaml(rule_file)
    s = res.resolve(Security(symbol=ES_CALL))
    assert s.settlement_type == SettlementType.CASH
    assert s.contract_multiplier == Decimal("1")


# --------------------------------------------------------------------------- #
# routing through the default chain
# --------------------------------------------------------------------------- #
def test_routes_through_default_resolver_chain() -> None:
    s = ResolverChain.default().resolve(Security(symbol=ES_CALL))
    assert s.security_id == "future_option:ES:U6:2026-07-01:call:6325"
    assert s.order_symbol == ES_CALL


def test_chain_still_resolves_other_families() -> None:
    chain = ResolverChain.default()
    assert (
        chain.resolve(
            Security(symbol="AAPL  250117C00150000", security_type=SecurityType.OPTION)
        ).security_id
        == "option:AAPL:2025-01-17:call:150"
    )
    assert (
        chain.resolve(Security(symbol="/ESM6", security_type=SecurityType.FUTURE)).security_id
        == "future:ES:M6"
    )
