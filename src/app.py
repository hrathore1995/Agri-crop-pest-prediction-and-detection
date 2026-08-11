"""
src/app.py  —  Agri decision-support suite (multi-tool).
Run with:  streamlit run src/app.py

Two tools, switch in the sidebar:
  1. Pest forecast (rice & cotton)  — weather -> next-week pest pressure (Track B)
  2. Plant health check (tomato)    — live sensor readings -> health status (Track A)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from xgboost import XGBRegressor

import pipeline as P

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "models" / "track_b_pooled_reg.json"
BUNDLE_PATH = ROOT / "models" / "app_bundle.pkl"
LEAF_MODEL   = ROOT / "models" / "multicrop_mobilenetv3.pt"
LEAF_CLASSES = ROOT / "models" / "multicrop_classes.json"
OOD_PATH     = ROOT / "models" / "multicrop_ood.npz"

st.set_page_config(page_title="Tatva Silicon Agri Decision Support", page_icon="🌱", layout="wide")
BAND_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


@st.cache_resource
def load_pest_artifacts():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    return model, bundle


def nice_label(cls):
    parts = cls.split("_", 1)
    crop = parts[0].capitalize()
    disease = " ".join(parts[1].replace("_", " ").split()) if len(parts) > 1 else ""
    return crop, (disease if disease and disease != "healthy" else "Healthy")


@st.cache_resource
def load_leaf_model():
    import torch, json
    from torchvision import models
    classes = json.load(open(LEAF_CLASSES))
    m = models.mobilenet_v3_small()
    m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, len(classes))
    m.load_state_dict(torch.load(LEAF_MODEL, map_location="cpu"))
    m.eval()
    extractor = torch.nn.Sequential(m.features, m.avgpool).eval()
    return m, classes, extractor


@st.cache_resource
def load_ood():
    import numpy as np
    if not OOD_PATH.exists():
        return None
    z = np.load(OOD_PATH, allow_pickle=True)
    return {"means": z["means"], "cov_inv": z["cov_inv"],
            "threshold": float(z["threshold"])}


def predict_leaf(model, classes, extractor, ood, pil_img, topk=3):
    import torch, numpy as np
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    x = tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        emb = extractor(x).flatten(1).numpy()[0]
    k = min(topk, len(classes))
    tp, ti = probs.topk(k)
    preds = [(classes[int(i)], float(p)) for p, i in zip(tp, ti)]
    score = None
    if ood is not None:
        diffs = emb[None, :] - ood["means"]
        score = float(np.einsum("cd,de,ce->c", diffs, ood["cov_inv"], diffs).min())
    return preds, score


# =====================================================================  PEST
def render_pest_page():
    try:
        model, bundle = load_pest_artifacts()
    except Exception as e:
        st.error("Could not load the pest model/bundle. Run "
                 "`python src/06_pooled_model.py` first.\n\n" + str(e))
        return
    series = pd.DataFrame(bundle["series_list"], columns=["crop","location","pest"])

    # ---- sidebar: what to forecast ----
    st.sidebar.header("1. What do you want to check?")
    crop = st.sidebar.selectbox("Crop", sorted(series.crop.unique()),
                                help="The crop you're growing.")
    locs = sorted(series[series.crop == crop].location.unique())
    location = st.sidebar.selectbox("Location (monitoring station)", locs,
                                    help="The research station nearest your area.")
    pests = sorted(series[(series.crop==crop) & (series.location==location)].pest.unique())
    pest = st.sidebar.selectbox("Pest / disease", pests,
                                help="The pest or disease you're worried about.")

    aw = bundle.get("active_weeks", {}).get((location, pest)) or list(range(1, 53))
    wmin, wmax = min(aw), max(aw)
    if len(aw) > 1:
        week = st.sidebar.select_slider(
            "Week of the year to forecast", options=aw, value=aw[len(aw)//2],
            help="1 = early January, 52 = late December. Only weeks when this pest "
                 "is actually monitored here are shown.")
    else:
        week = aw[0]
        st.sidebar.info(f"This pest is only monitored in week {aw[0]} here.")
    gap = "" if len(aw) == (wmax - wmin + 1) else \
        " · gaps in this range are weeks the pest isn't watched, and are skipped"
    st.sidebar.caption(f"This pest is monitored in weeks {wmin}–{wmax} at {location}{gap}.")

    st.sidebar.header("2. Weather scenario (optional)")
    st.sidebar.caption("These start at the **normal** weather for the week you picked. "
                       "Move them to ask *“what if it's warmer / wetter / more humid "
                       "than usual?”* and see how the pest outlook changes. Leave them "
                       "as-is for a normal-weather forecast.")
    offs = dict(
        temp_off=st.sidebar.slider(
            "Temperature vs normal (°C)", -5.0, 5.0, 0.0, 0.5,
            help="How many degrees warmer (+) or cooler (−) than usual for this week."),
        rain_mult=st.sidebar.slider(
            "Rainfall vs normal (×)", 0.0, 3.0, 1.0, 0.1,
            help="Multiply the usual rainfall. 1× = normal, 2× = twice as wet, 0× = dry."),
        humid_off=st.sidebar.slider(
            "Humidity vs normal (%)", -20.0, 20.0, 0.0, 1.0,
            help="Shift humidity above (+) or below (−) the usual level for this week."))

    # ---- main ----
    st.title("🌾 Pest Pressure Early-Warning Tool")
    st.caption("A weather-based early warning for crop pests · trained on the "
               "ICAR-CRIDA rice & cotton archive (real Indian field data)")

    with st.expander("ℹ️  New here? How to use this dashboard", expanded=True):
        st.markdown(
            "**What it does** — On the left, pick a crop, a location, and a pest. "
            "The tool estimates **how bad that pest is likely to be in the week you "
            "choose**, from the recent weather. Think of it as a weather forecast, "
            "but for pests.\n\n"
            "**How to read the answer**\n"
            "- **Predicted pest pressure** — the expected amount of the pest, in that "
            "pest's own measuring unit (shown in brackets). Bigger = more pest. It's "
            "an estimate, not a certainty.\n"
            "- **Risk band** — a simple traffic light for how heavy that is *for this "
            "pest, here*: 🟢 **Low** = barely present · 🟡 **Medium** = moderate · "
            "🔴 **High** = heavy, outbreak-level. (A “High” number for one pest can be "
            "very different from “High” for another — the tiers come from each pest's "
            "own history.)\n"
            "- **The chart** — the coloured line is the predicted level for every week "
            "of the season; the two flat lines mark where **Low → Medium** and "
            "**Medium → High** begin.\n\n"
            "**Tip:** the weather sliders on the left let you explore *what-if* "
            "conditions — try nudging temperature or rainfall up and watch the line move.")

    res = P.predict_severity(model, bundle, crop, location, pest, week, **offs)
    unit = bundle.get("units", {}).get((location, pest), "units")
    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    c1.metric(f"Predicted pest pressure", f"{res['count']:.0f}",
              help=f"Expected amount of the pest in week {week}, measured in "
                   f"“{unit}”. An estimate from the recent weather, not a guarantee.")
    c1.caption(f"measured in **{unit}**")
    c2.markdown(f"<div style='padding:0.4rem 0'>Risk band</div>"
                f"<div style='font-size:2rem;font-weight:700;"
                f"color:{BAND_COLORS[res['band']]}'>{res['band']}</div>",
                unsafe_allow_html=True)
    c2.caption("🟢 light · 🟡 moderate · 🔴 heavy")
    with c3:
        if res["endemic"]:
            st.warning("**Almost always around here.** This pest is present most of "
                       "the year, so the *time of year* matters more than the weather. "
                       "Treat the number as a rough seasonal guide.")
        else:
            st.success("**Comes and goes.** This pest appears in bursts that follow "
                       "the weather — the situation where this forecast helps most.")

    st.subheader(f"📅 Forecast across one typical season — {pest} at {location}")
    st.caption("X-axis = **week of the year (1–52)**, i.e. one single season. "
               "Y-axis = expected pest level. This is a what-if forecast for a "
               "typical year under the weather scenario you set on the left.")
    awset = set(aw)
    wfull = list(range(wmin, wmax + 1))
    counts = [P.predict_severity(model, bundle, crop, location, pest, w, **offs)["count"]
              if w in awset else np.nan          # break the line over gap weeks
              for w in wfull]
    prof = pd.DataFrame({"week": wfull, "predicted pressure": counts,
                         "medium threshold": res["q2"], "low threshold": res["q1"]}
                        ).set_index("week")
    st.line_chart(prof)
    st.caption(
        f"The **predicted pressure** line is the expected pest level each week (in "
        f"{unit}). The two flat lines are the risk-band cut-offs: **Low** up to "
        f"{res['q1']:.0f}, **Medium** up to {res['q2']:.0f}, **High** above that. "
        f"You picked week #{week}.")

    # ---- hidden-state estimation (Kalman filter on observed history) ----
    st.subheader("🔍 Real history & underlying trend (hidden-state estimate)")
    st.caption("Different from the chart above: X-axis here = **calendar year** "
               "(the real recorded history across many years), **not** week-of-year. "
               "Y-axis = pest level in the same unit.")
    os_ = bundle.get("obs_series", {}).get((location, pest))
    if os_ and len(os_[0]) >= 10:
        wk_ids = np.array(os_[0], dtype=float); vals = np.array(os_[1], dtype=float)
        est, lo, hi = P.kalman_smooth(wk_ids, vals)
        N = min(120, len(vals))
        sl = slice(-N, None)
        kdf = pd.DataFrame({"year": wk_ids[sl] / 52.0, "observed": vals[sl],
                            "estimated": est[sl], "lo": lo[sl], "hi": hi[sl]})
        band = alt.Chart(kdf).mark_area(opacity=0.18, color="#1f77b4").encode(
            x=alt.X("year:Q", title="calendar year",
                    axis=alt.Axis(format="d"), scale=alt.Scale(zero=False)),
            y=alt.Y("lo:Q", title=f"pest level ({unit})"), y2="hi:Q")
        dots = alt.Chart(kdf).mark_circle(size=16, color="#8a8a8a", opacity=0.7).encode(
            x="year:Q", y="observed:Q", tooltip=["year","observed"])
        line = alt.Chart(kdf).mark_line(color="#1f77b4", strokeWidth=2).encode(
            x="year:Q", y=alt.Y("estimated:Q"), tooltip=["year","estimated"])
        st.altair_chart((band + dots + line).properties(height=300),
                        use_container_width=True)
        st.caption(
            f"**How to read this:** the x-axis is the **calendar year**; each **grey "
            f"dot** is one real weekly count recorded that year (they jump around a lot "
            f"— that's measurement noise). The **blue line** is the estimated *true* "
            f"underlying pest pressure with the noise filtered out (the **hidden "
            f"state**). The **shaded band** is the uncertainty: about 95% of weekly "
            f"readings fall inside it, so a **narrow band means confident, a wide band "
            f"means unsure** (it widens across gaps in monitoring). Showing the most "
            f"recent {N} readings, in {unit}.")
    else:
        st.caption("Not enough observed history to estimate a hidden-state trend here.")

    with st.expander("How good is this forecast? (accuracy & limits)"):
        st.markdown(
            "- It forecasts next-week pest level from recent weather, learned from "
            "38 rice/cotton pest records across Indian stations (16,500+ weekly "
            "readings).\n"
            "- Tested on years it never saw during training (2006–2011), it clearly "
            "helps for **“comes-and-goes” pests** but adds little for **“always "
            "around” pests**, where the season already tells most of the story.\n"
            "- It's an **early-warning signal, not a precise count** — pest outbreaks "
            "are naturally noisy.\n"
            "- Covers **rice & cotton at specific Indian stations (1970s–2011)** only; "
            "it doesn't transfer to other crops or regions.")


# ===================================================================  TOMATO
def render_tomato_page():
    st.sidebar.header("Your greenhouse sensor readings")
    st.sidebar.caption("Enter what your sensors show right now.")
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 40.0, 25.0, 0.1,
        help="Current air temperature from your sensor.")
    humidity    = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 70.0, 0.5,
        help="Current air humidity from your sensor.")
    nitrogen    = st.sidebar.slider("Nitrogen (mg/kg)", 30.0, 130.0, 100.0, 1.0,
        help="Soil nitrogen level from the NPK sensor.")
    phosphorus  = st.sidebar.slider("Phosphorus (mg/kg)", 5.0, 120.0, 60.0, 1.0,
        help="Soil phosphorus level from the NPK sensor.")
    potassium   = st.sidebar.slider("Potassium (mg/kg)", 15.0, 150.0, 90.0, 1.0,
        help="Soil potassium level from the NPK sensor.")
    soil_raw    = st.sidebar.slider("Soil moisture (raw sensor)", 100, 700, 400, 1,
        help="Raw number from the soil-moisture sensor (not a %). Shown for reference.")

    readings = {"temperature_c": temperature, "humidity_pct": humidity,
                "nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium}
    res = P.tomato_health_check(readings)

    st.title("🍅 Tomato Plant-Health Check")
    st.caption("Checks live greenhouse sensor readings against tomato's healthy "
               "ranges · based on the Assam University real-sensor dataset")

    with st.expander("ℹ️  New here? How to use this dashboard", expanded=True):
        st.markdown(
            "**What it does** — Enter your greenhouse's **current sensor readings** on "
            "the left. The tool checks each one against the healthy range for tomato "
            "and tells you whether things look fine — and if not, exactly which "
            "reading is off.\n\n"
            "**How to read the answer**\n"
            "- **Overall status** — 🟢 **Healthy** = every reading is in its good "
            "range · 🔴 **At risk** = at least one reading is outside it.\n"
            "- **Readings outside healthy range** — how many of your inputs are out "
            "of the healthy zone.\n"
            "- **Per-variable check** — one row per reading: your value, the healthy "
            "range for tomato, and a ✓ (fine) or ✗ (out of range) so you know exactly "
            "what to adjust.")

    colour = "#2e7d32" if res["status"] == "Healthy" else "#c62828"
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"<div style='padding:0.4rem 0'>Overall status</div>"
                f"<div style='font-size:2.2rem;font-weight:700;color:{colour}'>"
                f"{res['status']}</div>", unsafe_allow_html=True)
    c2.metric("Readings outside healthy range", res["out_of_range"],
              help="How many of your sensor readings are outside tomato's healthy range.")

    df = pd.DataFrame(res["rows"])
    df["status"] = np.where(df["ok"], "✓ in range", "✗ out of range")
    st.subheader("Per-variable check")
    st.caption("Each row is one reading vs. the healthy range for tomato.")
    st.dataframe(df[["variable","value","healthy_range","status"]],
                 hide_index=True, use_container_width=True)
    st.caption(f"Soil moisture (raw): {soil_raw} — shown for reference only; a raw "
               "sensor value needs calibration before it can be range-checked.")

    with st.expander("What does this check really tell me? (important limits)"):
        st.markdown(
            "- It compares your readings against tomato's healthy ranges and flags "
            "what's outside them — a **status indicator, not a diagnosis**.\n"
            "- **Honest caveat:** the source dataset has **no recorded “healthy vs "
            "sick” labels** — “health” here simply *means* “inside these ranges.” So "
            "this is a transparent rule check, not a clever prediction.\n"
            "- **pH is left out** on purpose: that column in the dataset had no real "
            "sensor behind it and impossible values, so it wasn't trustworthy.\n"
            "- The ranges are sensible tomato starting points — double-check them "
            "against a horticulture reference before relying on this in a real "
            "greenhouse.")


# =====================================================================  LEAF
def render_leaf_page():
    st.sidebar.header("Upload a leaf photo")
    up = st.sidebar.file_uploader("Leaf image (jpg / png)",
                                  type=["jpg","jpeg","png"],
                                  help="A clear, close photo of a single leaf works best.")

    st.title("🍃 Leaf Disease Identifier")
    st.caption("Upload a leaf photo and the model names the most likely disease. "
               "Covers 5 crops: banana, cauliflower, chilli, groundnut, radish.")

    with st.expander("ℹ️  How to use this — and important limits", expanded=True):
        st.markdown(
            "**What it does** — Upload a leaf photo on the left; the model returns the "
            "most likely disease (or “healthy”) for that leaf, with a confidence score "
            "and the top-3 possibilities.\n\n"
            "**Only these 5 crops** — banana, cauliflower, chilli, groundnut, radish. "
            "For any other plant it will still force a guess, which won't mean anything.\n\n"
            "**Reality check (please read)** — the model scored ~96% on its *own* test "
            "images, but photos taken with a different phone, lighting, or background "
            "can do noticeably worse. Disease models often pick up on capture style, not "
            "just the lesion. **Treat this as an aid, not a diagnosis** — confirm anything "
            "important with an agronomist or a lab test. Low-confidence results especially "
            "should be taken with a pinch of salt.")

    if not (LEAF_MODEL.exists() and LEAF_CLASSES.exists()):
        st.warning("Leaf model not found. Train it first:  "
                   "`python src/11_train_multicrop.py`  (with QUICK = False).")
        return
    if up is None:
        st.info("⬅  Upload a leaf photo in the sidebar to get a prediction.")
        return

    strictness = st.sidebar.select_slider(
        "Non-leaf rejection", options=["Lenient","Balanced","Strict"], value="Balanced",
        help="How aggressively to reject images that don't look like a supported leaf. "
             "If your real leaf photos get wrongly rejected, choose 'Lenient'.")
    mult = {"Lenient": 1.3, "Balanced": 1.1, "Strict": 0.9}[strictness]

    from PIL import Image
    img = Image.open(up).convert("RGB")
    model, classes, extractor = load_leaf_model()
    ood = load_ood()
    preds, score = predict_leaf(model, classes, extractor, ood, img, topk=3)
    eff_thr = ood["threshold"] * mult if ood is not None else None
    is_ood = (score is not None) and (score > eff_thr)
    top_cls, top_conf = preds[0]
    crop, disease = nice_label(top_cls)

    c1, c2 = st.columns([1, 1.3])
    c1.image(img, caption="your photo", use_container_width=True)
    with c2:
        if is_ood:
            st.error("🚫 **This doesn't look like a supported leaf.** The image is "
                     "unlike anything the model was trained on, so it won't give a "
                     "reliable prediction. If it really is a banana, cauliflower, "
                     "chilli, groundnut, or radish leaf, try a clearer, closer, "
                     "well-lit photo of a single leaf.")
            with st.expander("Show the model's (unreliable) guess anyway"):
                st.write(f"{crop} — {disease} · {top_conf*100:.1f}% "
                         f"(ignore this — flagged as out-of-distribution)")
        else:
            st.markdown(f"### {crop} — {disease}")
            st.metric("Confidence", f"{top_conf*100:.1f}%")
            if top_conf < 0.5:
                st.warning("**Low confidence.** The photo may be unclear. Treat as a "
                           "rough guess.")
            st.caption("Top 3 possibilities:")
            for c, p in preds:
                cr, ds = nice_label(c)
                st.write(f"**{cr} — {ds}** · {p*100:.1f}%")
                st.progress(min(1.0, p))
    if ood is None:
        st.caption("Note: the 'not a leaf' detector isn't built yet — run "
                   "`python src/12_build_ood.py` to enable rejection of non-leaf images.")
    elif score is not None:
        st.caption(f"Out-of-distribution distance: **{score:.0f}** "
                   f"(rejected above **{eff_thr:.0f}** at '{strictness}' setting). "
                   f"Lower = more leaf-like. Adjust the strictness slider if real "
                   f"leaves are wrongly rejected or non-leaves slip through.")
    st.caption("Reminder: an aid, not a diagnosis, even when confident.")


# =====================================================================  ROUTER
st.sidebar.title("🌱 Agri Decision Support")
tool = st.sidebar.radio(
    "Choose a tool",
    ["Pest forecast (rice & cotton)",
     "Plant health check (tomato)",
     "Leaf disease ID (photo)"],
    help="Three separate tools — pick one.")
st.sidebar.divider()
if tool.startswith("Pest"):
    render_pest_page()
elif tool.startswith("Plant"):
    render_tomato_page()
else:
    render_leaf_page()