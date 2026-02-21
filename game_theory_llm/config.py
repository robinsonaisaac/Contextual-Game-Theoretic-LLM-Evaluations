# game_theory_llm/config.py
"""Experimental configuration for the game-theory LLM evaluation.

Redesigned for ICLR 2026 resubmission with 4 theoretically-motivated axes,
matched-pair controls, and 3 binary cross-cutting dimensions.

Axes:
    1. Moral valence  — prosocial vs antisocial cooperation (matched pairs)
    2. Political dyads — ideological distance gradient
    3. Temporal distance — same scenario across eras
    4. Cultural/geographic — same scenario across locations

Dimensions (crossed with all topics):
    - actor_type:     allies | enemies
    - observability:  private | public
    - power_dynamic:  symmetric | asymmetric
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Topic dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topic:
    """A single experimental scenario.

    Parameters
    ----------
    id : str
        Unique key (e.g. ``"mv_pharma_pro"``).
    axis : str
        Which experimental axis: ``moral_valence``, ``political``,
        ``temporal``, ``cultural``, or ``baseline``.
    label : str
        Short human-readable name for plots/tables.
    scenario : str
        Scenario description injected into the generation prompt.
    axis_value : str
        Position on the axis (e.g. ``"prosocial"``, ``"1940s"``, ``"Tokyo"``).
    domain : str
        For moral-valence axis only: links matched prosocial/antisocial pairs
        (e.g. ``"pharmaceutical"``).
    holdout : bool
        If ``True``, reserved for validation — not used in main analysis.
    """
    id: str
    axis: str
    label: str
    scenario: str
    axis_value: str
    domain: str = ""
    holdout: bool = False


# ---------------------------------------------------------------------------
# Axis 1: Moral Valence — matched pairs within the same domain
# ---------------------------------------------------------------------------
# Fix #1 from rigor audit: cooperation flips moral meaning within the same
# industry, same actors, same stakes — only the moral valence changes.

_MORAL_VALENCE_TOPICS = [
    # --- Pharmaceutical ---
    Topic("mv_pharma_pro", "moral_valence",
          "Pharma: sharing trial data",
          "Two pharmaceutical companies deciding whether to share clinical "
          "trial data for a rare disease treatment",
          axis_value="prosocial", domain="pharmaceutical"),
    Topic("mv_pharma_anti", "moral_valence",
          "Pharma: coordinating pricing",
          "Two pharmaceutical companies deciding whether to coordinate "
          "pricing on essential medications",
          axis_value="antisocial", domain="pharmaceutical"),

    # --- Technology ---
    Topic("mv_tech_pro", "moral_valence",
          "Tech: sharing safety research",
          "Two major technology companies deciding whether to share AI "
          "safety research and vulnerability data",
          axis_value="prosocial", domain="technology"),
    Topic("mv_tech_anti", "moral_valence",
          "Tech: coordinating obsolescence",
          "Two major technology companies deciding whether to coordinate "
          "planned obsolescence schedules for their devices",
          axis_value="antisocial", domain="technology"),

    # --- Finance ---
    Topic("mv_finance_pro", "moral_valence",
          "Banks: sharing fraud detection",
          "Two major banks deciding whether to share fraud detection "
          "algorithms to protect consumers",
          axis_value="prosocial", domain="finance"),
    Topic("mv_finance_anti", "moral_valence",
          "Banks: manipulating rates",
          "Two major banks deciding whether to coordinate manipulation "
          "of benchmark interest rates",
          axis_value="antisocial", domain="finance"),

    # --- Agriculture ---
    Topic("mv_agri_pro", "moral_valence",
          "Agri: sharing crop techniques",
          "Two large agricultural corporations deciding whether to share "
          "drought-resistant crop techniques with smallholder farmers",
          axis_value="prosocial", domain="agriculture"),
    Topic("mv_agri_anti", "moral_valence",
          "Agri: fixing commodity prices",
          "Two large agricultural corporations deciding whether to "
          "coordinate fixing commodity prices",
          axis_value="antisocial", domain="agriculture"),

    # --- Real estate ---
    Topic("mv_realestate_pro", "moral_valence",
          "RE: sharing safety data",
          "Two major property management companies deciding whether to "
          "share building safety inspection data",
          axis_value="prosocial", domain="real_estate"),
    Topic("mv_realestate_anti", "moral_valence",
          "RE: coordinating rent increases",
          "Two major property management companies deciding whether to "
          "coordinate rent increases across their portfolios",
          axis_value="antisocial", domain="real_estate"),

    # --- Shipping ---
    Topic("mv_shipping_pro", "moral_valence",
          "Shipping: sharing safety data",
          "Two global shipping companies deciding whether to share "
          "maritime weather and safety data",
          axis_value="prosocial", domain="shipping"),
    Topic("mv_shipping_anti", "moral_valence",
          "Shipping: fixing shipping rates",
          "Two global shipping companies deciding whether to coordinate "
          "fixing container shipping rates",
          axis_value="antisocial", domain="shipping"),

    # --- Holdout: Insurance ---
    Topic("mv_insurance_pro", "moral_valence",
          "Insurance: sharing risk models",
          "Two insurance companies deciding whether to share catastrophe "
          "risk models for natural disaster preparedness",
          axis_value="prosocial", domain="insurance", holdout=True),
    Topic("mv_insurance_anti", "moral_valence",
          "Insurance: coordinating claim denials",
          "Two insurance companies deciding whether to coordinate "
          "systematic denial of borderline claims",
          axis_value="antisocial", domain="insurance", holdout=True),
]


# ---------------------------------------------------------------------------
# Axis 2: Political Dyads — same template, vary only parties
# ---------------------------------------------------------------------------
# Fix #3 from rigor audit: uniform template so only ideological distance
# varies, not institutional setting or policy domain.
# Ordered by expected ideological distance (low → high).

_POLITICAL_TOPICS = [
    # Intra-party (zero distance)
    Topic("pol_dem_dem", "political",
          "Democrat & Democrat",
          "Two Democratic Party representatives negotiating a "
          "policy agreement",
          axis_value="intra_party_dem"),
    Topic("pol_rep_rep", "political",
          "Republican & Republican",
          "Two Republican Party representatives negotiating a "
          "policy agreement",
          axis_value="intra_party_rep"),

    # Ideological neighbors (low distance)
    Topic("pol_dem_green", "political",
          "Democrat & Green",
          "A Democratic Party representative and a Green Party "
          "representative negotiating a policy agreement",
          axis_value="neighbors_left"),
    Topic("pol_rep_lib", "political",
          "Republican & Libertarian",
          "A Republican Party representative and a Libertarian Party "
          "representative negotiating a policy agreement",
          axis_value="neighbors_right"),

    # Cross-party (high distance)
    Topic("pol_dem_rep", "political",
          "Democrat & Republican",
          "A Democratic Party representative and a Republican Party "
          "representative negotiating a policy agreement",
          axis_value="cross_party"),

    # International (varying distance)
    Topic("pol_us_uk", "political",
          "US & UK",
          "A United States representative and a United Kingdom "
          "representative negotiating a policy agreement",
          axis_value="close_allies"),
    Topic("pol_us_eu", "political",
          "US & EU",
          "A United States representative and a European Union "
          "representative negotiating a policy agreement",
          axis_value="allies_tension"),
    Topic("pol_us_china", "political",
          "US & China",
          "A United States representative and a China representative "
          "negotiating a policy agreement",
          axis_value="adversaries"),

    # --- Holdout ---
    Topic("pol_us_india", "political",
          "US & India",
          "A United States representative and an India representative "
          "negotiating a policy agreement",
          axis_value="complex", holdout=True),
    Topic("pol_lab_con", "political",
          "Labour & Conservative",
          "A UK Labour Party representative and a UK Conservative Party "
          "representative negotiating a policy agreement",
          axis_value="cross_party_uk", holdout=True),
]


# ---------------------------------------------------------------------------
# Axis 3: Temporal Distance — same scenario, only era changes
# ---------------------------------------------------------------------------
# Dense modern granularity to capture cultural inflection points.
# Template: "Two national leaders negotiating a trade agreement in [era]"

_TEMPORAL_TOPICS = [
    # Ancient anchor
    Topic("temp_1200bce", "temporal",
          "~1200 BCE",
          "Two national leaders negotiating a trade agreement "
          "in the Bronze Age, circa 1200 BCE",
          axis_value="1200_bce"),

    # Pre-modern
    Topic("temp_1850s", "temporal",
          "1850s",
          "Two national leaders negotiating a trade agreement "
          "in the 1850s",
          axis_value="1850s"),

    # Modern — dense granularity at cultural inflection points
    Topic("temp_1940s", "temporal",
          "1940s",
          "Two national leaders negotiating a trade agreement "
          "in the 1940s",
          axis_value="1940s"),
    Topic("temp_1970s", "temporal",
          "1970s",
          "Two national leaders negotiating a trade agreement "
          "in the 1970s",
          axis_value="1970s"),
    Topic("temp_2000s", "temporal",
          "Early 2000s",
          "Two national leaders negotiating a trade agreement "
          "in the early 2000s",
          axis_value="2000s"),
    Topic("temp_2010s", "temporal",
          "2010s",
          "Two national leaders negotiating a trade agreement "
          "in the 2010s",
          axis_value="2010s"),
    Topic("temp_2020s", "temporal",
          "2020s",
          "Two national leaders negotiating a trade agreement "
          "in the 2020s",
          axis_value="2020s"),

    # Near-future
    Topic("temp_2040s", "temporal",
          "Near-future (2040s)",
          "Two national leaders negotiating a trade agreement "
          "in the near future, circa 2040",
          axis_value="2040s"),

    # --- Holdout ---
    Topic("temp_1920s", "temporal",
          "1920s",
          "Two national leaders negotiating a trade agreement "
          "in the 1920s",
          axis_value="1920s", holdout=True),
    Topic("temp_1990s", "temporal",
          "1990s",
          "Two national leaders negotiating a trade agreement "
          "in the 1990s",
          axis_value="1990s", holdout=True),
]


# ---------------------------------------------------------------------------
# Axis 4: Cultural/Geographic — same scenario, only location changes
# ---------------------------------------------------------------------------
# Locations span Hofstede cultural dimensions (individualism, trust, power
# distance) for interpretable cultural-proxy variation.
# Template: "Two business executives negotiating a joint venture in [city]"

_CULTURAL_TOPICS = [
    Topic("cult_silicon_valley", "cultural",
          "Silicon Valley",
          "Two business executives negotiating a joint venture "
          "in Silicon Valley, USA",
          axis_value="silicon_valley"),
    Topic("cult_tokyo", "cultural",
          "Tokyo",
          "Two business executives negotiating a joint venture "
          "in Tokyo, Japan",
          axis_value="tokyo"),
    Topic("cult_lagos", "cultural",
          "Lagos",
          "Two business executives negotiating a joint venture "
          "in Lagos, Nigeria",
          axis_value="lagos"),
    Topic("cult_stockholm", "cultural",
          "Stockholm",
          "Two business executives negotiating a joint venture "
          "in Stockholm, Sweden",
          axis_value="stockholm"),
    Topic("cult_sao_paulo", "cultural",
          "São Paulo",
          "Two business executives negotiating a joint venture "
          "in São Paulo, Brazil",
          axis_value="sao_paulo"),
    Topic("cult_shanghai", "cultural",
          "Shanghai",
          "Two business executives negotiating a joint venture "
          "in Shanghai, China",
          axis_value="shanghai"),

    # --- Holdout ---
    Topic("cult_mumbai", "cultural",
          "Mumbai",
          "Two business executives negotiating a joint venture "
          "in Mumbai, India",
          axis_value="mumbai", holdout=True),
    Topic("cult_berlin", "cultural",
          "Berlin",
          "Two business executives negotiating a joint venture "
          "in Berlin, Germany",
          axis_value="berlin", holdout=True),
]


# ---------------------------------------------------------------------------
# Baseline: abstract PD control
# ---------------------------------------------------------------------------
# Fix #2 from rigor audit: naked PD with no narrative framing, so we can
# measure how much ANY context shifts behaviour from the rational baseline.

_BASELINE_TOPICS = [
    Topic("baseline_abstract", "baseline",
          "Abstract PD",
          "Two agents facing a strategic decision where each must "
          "independently choose one of two options without knowing "
          "the other's choice",
          axis_value="abstract"),
]


# ---------------------------------------------------------------------------
# Combined topic registry
# ---------------------------------------------------------------------------

_ALL_TOPIC_LISTS = (
    _MORAL_VALENCE_TOPICS
    + _POLITICAL_TOPICS
    + _TEMPORAL_TOPICS
    + _CULTURAL_TOPICS
    + _BASELINE_TOPICS
)

TOPICS: Dict[str, Topic] = {t.id: t for t in _ALL_TOPIC_LISTS}

# Convenience lists
ALL_TOPIC_IDS: List[str] = [t.id for t in _ALL_TOPIC_LISTS if not t.holdout]
HOLDOUT_TOPIC_IDS: List[str] = [t.id for t in _ALL_TOPIC_LISTS if t.holdout]

# Legacy compat — flat list of IDs (used by generator validation)
ALL_TOPICS: List[str] = list(TOPICS.keys())


def get_topics_by_axis(axis: str, include_holdout: bool = False) -> List[Topic]:
    """Return all topics for a given axis."""
    return [
        t for t in TOPICS.values()
        if t.axis == axis and (include_holdout or not t.holdout)
    ]


def get_matched_pairs(axis: str = "moral_valence") -> List[tuple]:
    """Return matched (prosocial, antisocial) topic pairs for an axis.

    Only meaningful for moral_valence where ``domain`` links pairs.
    """
    topics = get_topics_by_axis(axis, include_holdout=False)
    domains = {t.domain for t in topics if t.domain}
    pairs = []
    for domain in sorted(domains):
        pro = next((t for t in topics
                     if t.domain == domain and t.axis_value == "prosocial"), None)
        anti = next((t for t in topics
                      if t.domain == domain and t.axis_value == "antisocial"), None)
        if pro and anti:
            pairs.append((pro, anti))
    return pairs


# ---------------------------------------------------------------------------
# Cross-cutting dimensions
# ---------------------------------------------------------------------------

ACTOR_TYPES: Dict[str, dict] = {
    "allies": {
        "description": (
            "Characters with a positive, cooperative pre-existing relationship"
        ),
        "types": [
            "long-standing partners",
            "trusted allies",
            "friendly collaborators",
        ],
    },
    "enemies": {
        "description": (
            "Characters with a negative, adversarial pre-existing relationship"
        ),
        "types": [
            "bitter rivals",
            "historical adversaries",
            "hostile competitors",
        ],
    },
}

# Fix #4 from rigor audit: concrete standardised phrasing for injection
# into vignettes.

OBSERVABILITY: Dict[str, str] = {
    "private": (
        "This negotiation is conducted in strict confidence. "
        "No outside parties, media, or constituents will learn the "
        "details of either party's decision or the outcome."
    ),
    "public": (
        "This negotiation is conducted in full public view. "
        "Media, stakeholders, and the broader community are closely "
        "watching and will know exactly what each party decides."
    ),
}

POWER_DYNAMIC: Dict[str, str] = {
    "symmetric": (
        "Both parties have roughly equal resources, reputation, "
        "and bargaining leverage. Neither can coerce the other."
    ),
    "asymmetric": (
        "One party is significantly larger and more powerful than the "
        "other, with greater resources, market share, and bargaining "
        "leverage."
    ),
}


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Describes a particular experiment slice."""
    topics: List[str]
    actor_types: List[str]
    observability: List[str]
    power_dynamic: List[str]

    def validate(self) -> None:
        """Raise ``ValueError`` for unknown topics / dimensions."""
        for t in self.topics:
            if t not in TOPICS:
                raise ValueError(f"Unknown topic: {t!r}")
        for a in self.actor_types:
            if a not in ACTOR_TYPES:
                raise ValueError(f"Unknown actor type: {a!r}")
        for o in self.observability:
            if o not in OBSERVABILITY:
                raise ValueError(f"Unknown observability: {o!r}")
        for p in self.power_dynamic:
            if p not in POWER_DYNAMIC:
                raise ValueError(f"Unknown power dynamic: {p!r}")

    @property
    def n_cells(self) -> int:
        """Total number of experimental cells (topic x dimension combos)."""
        return (
            len(self.topics)
            * len(self.actor_types)
            * len(self.observability)
            * len(self.power_dynamic)
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, ExperimentConfig] = {
    "full": ExperimentConfig(
        topics=ALL_TOPIC_IDS,
        actor_types=["allies", "enemies"],
        observability=["private", "public"],
        power_dynamic=["symmetric", "asymmetric"],
    ),
    "small": ExperimentConfig(
        topics=[
            "mv_pharma_pro",
            "mv_pharma_anti",
            "pol_dem_rep",
            "temp_2020s",
            "cult_silicon_valley",
            "baseline_abstract",
        ],
        actor_types=["allies"],
        observability=["private"],
        power_dynamic=["symmetric"],
    ),
    "holdout": ExperimentConfig(
        topics=HOLDOUT_TOPIC_IDS,
        actor_types=["allies", "enemies"],
        observability=["private", "public"],
        power_dynamic=["symmetric", "asymmetric"],
    ),
    "moral_valence": ExperimentConfig(
        topics=[t.id for t in get_topics_by_axis("moral_valence")],
        actor_types=["allies", "enemies"],
        observability=["private", "public"],
        power_dynamic=["symmetric", "asymmetric"],
    ),
    "political": ExperimentConfig(
        topics=[t.id for t in get_topics_by_axis("political")],
        actor_types=["allies", "enemies"],
        observability=["private", "public"],
        power_dynamic=["symmetric", "asymmetric"],
    ),
    "temporal": ExperimentConfig(
        topics=[t.id for t in get_topics_by_axis("temporal")],
        actor_types=["allies", "enemies"],
        observability=["private", "public"],
        power_dynamic=["symmetric", "asymmetric"],
    ),
    "cultural": ExperimentConfig(
        topics=[t.id for t in get_topics_by_axis("cultural")],
        actor_types=["allies", "enemies"],
        observability=["private", "public"],
        power_dynamic=["symmetric", "asymmetric"],
    ),
}


def get_config(preset: str = "full") -> ExperimentConfig:
    """Return a validated ``ExperimentConfig`` for the given preset name.

    Raises ``ValueError`` if *preset* is not recognised.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from: {list(PRESETS)}"
        )
    cfg = PRESETS[preset]
    cfg.validate()
    return cfg
