import os, requests
import geocoder

# Fields
BASE_API_URL = "https://file.tellusxdp.com/api/v1/origin/search/" # https://www.tellusxdp.com/docs/api-reference/palsar2-files-v1.html#/
ACCESS_TOKEN = "YOUR_TOKEN"
HEADERS = {"Authorization": "Bearer " + ACCESS_TOKEN}
TARGET_PLACE="Skytree, Tokyo"
SAVE_DIRECTORY="./data/"

# Functions
def rect_vs_point(ax, ay, aw, ah, bx, by):
    return 1 if bx > ax and bx < aw and by > ay and by < ah else 0
def get_scene_list_slc(_get_params={}):
    query = "palsar2-l11"
    r = requests.get(BASE_API_URL + query, _get_params, headers=HEADERS)
    if not r.status_code == requests.codes.ok:
        r.raise_for_status()
    return r.json()

def get_slc_scenes(_target_json, _get_params={}):
    # get file list
    r = requests.get(_target_json["publish_link"], _get_params, headers=HEADERS)
    if not r.status_code == requests.codes.ok:
        r.raise_for_status()
    file_list = r.json()['files']
    dataset_id = _target_json['dataset_id'] # folder name
    # make dir
    if os.path.exists(SAVE_DIRECTORY + dataset_id) == False:
        os.makedirs(SAVE_DIRECTORY + dataset_id)
    # downloading
    print("[Start downloading]", dataset_id)
    for _tmp in file_list:
        r = requests.get(_tmp['url'], headers=HEADERS, stream=True)
        if not r.status_code == requests.codes.ok:
            r.raise_for_status()
        with open(os.path.join(SAVE_DIRECTORY, dataset_id, _tmp['file_name']), "wb") as f:
            f.write(r.content)
        print("  - [Done]", _tmp['file_name'])
    print("finished")
    return

# Entry point
def main():
    # extract slc list around the address
    gc = geocoder.osm(TARGET_PLACE, timeout=5.0) # get latlon
    slc_list_json = get_scene_list_slc({'after':'2019-1-01T00:00:00', 'polarisations':'HH+HV', 'page_size':'1000'})  # get single look complex list
    target_places_json = [_ for _ in slc_list_json['items'] if rect_vs_point(_['bbox'][1], _['bbox'][0], _['bbox'][3], _['bbox'][2], gc.latlng[0], gc.latlng[1])] # lot_min, lat_min, lot_max...
    target_ids = [_['dataset_id'] for _ in target_places_json]
    print("[Matched SLCs]", target_ids)
    for target_id in target_ids:
        target_json = [_ for _ in slc_list_json['items'] if _['dataset_id'] == target_id][0]
        # download the target file
        get_slc_scenes(target_json)
        break # if remove, download all data
    
if __name__=="__main__":
       main()

