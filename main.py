from pathlib import Path
import pandas as pd

# Set pandas display options to show all columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def derive_filename(data_pack: Path):
    return {
        "abnormal_log": data_pack / "abnormal_logs.parquet",
        "normal_log": data_pack / "normal_logs.parquet",
        "abnormal_metric": data_pack / "abnormal_metrics.parquet",
        "normal_metric": data_pack / "normal_metrics.parquet",
        "abnormal_trace": data_pack / "abnormal_traces.parquet",
        "normal_trace": data_pack / "normal_traces.parquet",
        "env": data_pack / "env.json",
        "injection": data_pack / "injection.json",
    }


def main():
    cases = pd.read_parquet(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
    )
    print(cases.columns)
    top_10 = cases["datapack"].head(10).tolist()

    data_paths = [
        Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/{i}")
        for i in top_10
    ]

    d = derive_filename(data_paths[0])
    print("=== Normal Metrics ===")
    print(pd.read_parquet(d["normal_metric"]).head(1))
    print("\n=== Normal Logs ===")
    print(pd.read_parquet(d["normal_log"]).head(1))
    print("\n=== Normal Traces ===")
    print(pd.read_parquet(d["normal_trace"]).head(1))

    """
=== Normal Metrics ===
                                 time                              metric  value service_name attr.k8s.node.name attr.k8s.namespace.name attr.k8s.statefulset.name attr.k8s.deployment.name attr.k8s.replicaset.name                       attr.k8s.pod.name attr.k8s.container.name
0 2025-06-14 13:16:53.775440694+00:00  container.memory.major_page_faults    0.0         None            worker4                     ts2                      None    ts-route-plan-service                     None  ts-route-plan-service-76b455cf6c-bh57r   ts-route-plan-service

=== Normal Logs ===
                              time                          trace_id           span_id level       service_name                                                    message                  attr.k8s.pod.name attr.k8s.service.name attr.k8s.namespace.name
0 2025-06-14 13:16:53.982000+00:00  eb52200932ecb1571f8cc089dd31cf9f  d8b44231c2724267  INFO  ts-config-service  Initializing Spring DispatcherServlet 'dispatcherServlet'  ts-config-service-d488d7cfd-tnjhq     ts-config-service                     ts2

=== Normal Traces ===
                              time                          trace_id           span_id    parent_span_id                                       span_name attr.span_kind       service_name    duration attr.status_code                  attr.k8s.pod.name attr.k8s.service.name attr.k8s.namespace.name  attr.http.request.content_length  attr.http.response.content_length attr.http.request.method  attr.http.response.status_code
0 2025-06-14 13:16:53.900000+00:00  249760c9ec10a5f8a4166739e21e280a  f35933ae6ca1b901  10af6183ff288c0e  GET /api/v1/configservice/configs/{configName}         Server  ts-config-service  1166528342            Unset  ts-config-service-d488d7cfd-tnjhq     ts-config-service                     ts2                               NaN                                NaN                      GET                           200.0
    """


if __name__ == "__main__":
    main()
