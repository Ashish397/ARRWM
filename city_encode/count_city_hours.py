import zarr, os, json, glob

ts_to_city = {}
for fpath in glob.glob("/projects/u6ex/fbots/frodobots_captions/train/output_rides_*/ride_*/ride_*_video_captions_OpenGVLab_InternVL3_8B.json"):
    try:
        with open(fpath) as fh:
            d = json.load(fh)
        loc = d.get("metadata", {}).get("location", "")
        if loc in ("Madrid", "Rome", "Stockholm"):
            ride_name = fpath.split("/")[-2]
            ts = ride_name.split("_")[-1]
            ts_to_city[ts] = loc
    except:
        pass

encoded_dir = "/projects/u6ex/fbots/frodobots_encoded"
city_stats = {}
for zarr_name in sorted(os.listdir(encoded_dir)):
    if not zarr_name.endswith(".zarr"):
        continue
    ts = zarr_name.replace(".zarr", "")
    city = ts_to_city.get(ts)
    if city is None:
        continue
    try:
        g = zarr.open_group(os.path.join(encoded_dir, zarr_name), mode="r")
        n_latents = g["latents"].shape[0]
        dur_sec = n_latents * 4 / 16.0
        city_stats.setdefault(city, {"count": 0, "hours": 0.0})
        city_stats[city]["count"] += 1
        city_stats[city]["hours"] += dur_sec / 3600
    except:
        pass

for city in sorted(city_stats):
    s = city_stats[city]
    print("%s: %d zarrs, %.1fh" % (city, s["count"], s["hours"]))
total_h = sum(s["hours"] for s in city_stats.values())
total_c = sum(s["count"] for s in city_stats.values())
print("Total: %d zarrs, %.1fh" % (total_c, total_h))
