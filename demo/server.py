# coding: utf-8

import json
import falcon
import sys
from statsd import StatsClient

from hmtlPredictor import HMTLPredictor

STATSD = StatsClient()


class AllResource(object):
    def __init__(self, model_name="conll_full_elmo", mode="demo"):
        self.jmd = HMTLPredictor(model_name=model_name)
        self.mode = mode
        print(f"Server loaded with model {model_name}")
        self.response = None

    def on_get(self, req, resp):
        self.response = {}
        text = req.get_param("text")  # Input text
        STATSD.incr(f"huggingNLP-{self.mode}.msg")

        raw_format = req.get_param_as_bool("raw", required=False, blank_as_true=False)  # Non-formatted output
        raw_format = False if raw_format is None else raw_format

        self.response["text"] = text

        if text is not None:
            STATSD.incr(f"huggingNLP-{self.mode}.not-empty-msg")
            with STATSD.timer(f"huggingNLP-{self.mode}.timing.inference"):
                message_filtered, model_prediction = self.jmd.predict(text, raw_format=raw_format)
                for key, value in model_prediction.items():
                    self.response[key] = value

                if message_filtered:
                    STATSD.incr(f"huggingNLP-{self.mode}.filtered-msg")
        else:
            STATSD.incr(f"huggingNLP-{self.mode}.empty-msg")

        resp.body = json.dumps(self.response)
        resp.content_type = "application/json"
        resp.append_header("Access-Control-Allow-Origin", "*")
        resp.status = falcon.HTTP_200


def build_app(model_name="conll_full_elmo", mode="demo"):
    APP = falcon.API()
    APP.req_options.auto_parse_qs_csv = False
    ALL_RESOURCE = AllResource(model_name=model_name, mode=mode)
    APP.add_route("/jmd", ALL_RESOURCE)
    return APP
