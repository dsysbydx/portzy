import json
from portfolio import Portfolio
import traceback

def handle(request):

    # parse request
    request_json = request.get_json(silent=True)
    request_args = request.args
    if request_json and 'gs_url' in request_json:
        gs_url = request_json['gs_url']
    elif request_args and 'gs_url' in request_args:
        gs_url = request_args['gs_url']
    else:
        print(traceback.format_exc())
        raise ValueError("JSON is invalid, or missing a 'gs_url' property")
    p = Portfolio.from_gs(gs_url)
    res = {'gsURL': gs_url}
    res.update(p.get_summary())
    res_js = json.dumps(res)
    return res_js, 200, {'ContentType': 'application/json'}
