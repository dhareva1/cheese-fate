"""Organism-specific starting parameters for the mechanistic baseline, with provenance.

Cardinal values are fixed during fitting unless a sensitivity analysis frees them. Each
preset records where every fixed value comes from and whether that source was checked
against the original publication. Values marked provisional must be confirmed from the full
text before they are reported in the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from cheese_fate.mechanistic import Parameters

# headroom (growth ceiling relative to the starting count) is fitted because pathogen maxima
# in cheese (median about 4-5 log CFU/g) sit far below broth maxima and vary by batch,
# consistent with suppression by LAB; a single absolute ceiling forced high-start curves
# down. z_thermal is fitted because heat kill in curd is much weaker than a broth z-value
# implies (observed -1 to -4 log across hot steps).
# log_concentration is fitted because counts rise by a median of about 1.3-1.9 log within
# the first day even when the curd is cooked above 48 C, the signature of curd concentration.
FREE = (
    "mu_opt", "h0", "headroom", "log_concentration", "log_d_ref", "z_ph", "log_d_thermal",
    "z_thermal",
)
BOUNDS = {
    "log_concentration": (0.0, 2.0),
    "mu_opt": (0.01, 5.0),
    "h0": (0.0, 30.0),
    "headroom": (0.0, 8.0),
    "log_d_ref": (0.0, 5.0),
    "z_ph": (0.1, 10.0),
    "log_d_thermal": (-4.0, 3.0),
    "z_thermal": (3.0, 40.0),
}


@dataclass(frozen=True)
class Preset:
    organism: str
    params: Parameters
    sources: dict[str, str]


PRESETS = {
    "Staphylococcus aureus": Preset(
        organism="Staphylococcus aureus",
        params=Parameters(
            t_min=7.72, t_opt=40.63, t_max=46.73, aw_min=0.808,
            ph_min=4.2, ph_opt=7.0, ph_max=9.3, headroom=2.0,
        ),
        sources={
            "t_min, t_opt, t_max, aw_min": (
                "Medved'ova A, Havlikova A, Lehotova V, Valik L (2019). Italian Journal of Food "
                "Safety 8:8287. doi:10.4081/ijfs.2019.8287 (cardinal model, broth; verified)"
            ),
            "ph_min, ph_opt, ph_max": "commonly cited growth limits; PROVISIONAL, source not verified",
        },
    ),
    "Escherichia coli": Preset(
        organism="Escherichia coli",
        params=Parameters(
            t_min=-1.27, t_opt=37.26, t_max=45.12, aw_min=0.95,
            ph_min=4.30, ph_opt=7.09, ph_max=9.48, headroom=2.0,
        ),
        sources={
            "t_min, t_opt, t_max, ph_min, ph_opt": (
                "Baka M, Van Derlinden E, Boons K, Mertens L, Van Impe JF (2013). Food Control "
                "29:328-335, doi:10.1016/j.foodcont.2012.04.022 (E. coli K12 MG1655, CTMI/CPM): "
                "Tmin -1.273 (SE 0.179), Topt 37.26 (SE 0.688), Tmax 45.12 (SE 0.013), pHmin "
                "4.303 (SE 0.014), pHopt 7.085; matches the article's abstract as indexed "
                "2026-10-05; full text (paywalled) not checked"
            ),
            "ph_max": "attributed to Baka et al. (2013) by a secondary source; PROVISIONAL",
            "aw_min": "commonly cited minimum aw for E. coli growth; PROVISIONAL",
        },
    ),
}
