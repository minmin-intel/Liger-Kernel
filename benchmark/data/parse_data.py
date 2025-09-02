import pandas as pd

def sample(csv_file):
    df = pd.read_csv(csv_file)
    df_sampled = df.loc[(df["kernel_name"]=="rope") & (df["kernel_provider"]=="liger")]
    print(df_sampled.shape[0])
    df_sampled.to_csv("rope_liger_benchmark.csv", index=False)


# SPEED_PRINT_TEMPLATE="""\
# Benchmark conditions: {condition}
# Execution latency of {kernel_name} in {kernel_operation_mode}:
# * {min_latency} ms: when {x_name} = {min_x_value}
# * {max_latency} ms: when {x_name} = {max_x_value}
# """

SPEED_PRINT_TEMPLATE="""\
Average execution latency of {kernel_name} 
* forward pass: {avg_fwd_latency:.3f} ms
* backward pass: {avg_bwd_latency:.3f} ms
"""

# MEMORY_PRINT_TEMPLATE="""\
# Memory usage of {kernel_name}:
# * {min_memory} MB: when {x_name} = {min_x_value}
# * {max_memory} MB: when {x_name} = {max_x_value}
# """
MEMORY_PRINT_TEMPLATE="""\
Average memory usage of {kernel_name}: {avg_memory:.0f} MB
"""

def parse_benchmark_output(csv_file: str):
    df = pd.read_csv(csv_file)

    metric_names = ["speed", "memory"]
    metrics = {
        "fwd_ms": None,
        "bwd_ms": None,
        "memory": None
    }

    for metric in metric_names:
        if metric == "speed":
            
            df_metric = df.loc[(df["metric_name"]==metric) & (df["kernel_operation_mode"]=="forward")]
            metrics["fwd_ms"] = df_metric["y_value_50"].mean()
            df_metric = df.loc[(df["metric_name"]==metric) & (df["kernel_operation_mode"]=="backward")]
            metrics["bwd_ms"] = df_metric["y_value_50"].mean()

            print(SPEED_PRINT_TEMPLATE.format(
                kernel_name="rope",
                avg_fwd_latency=metrics["fwd_ms"],
                avg_bwd_latency=metrics["bwd_ms"]
            ))
        elif metric == "memory":
            df_metric = df.loc[(df["metric_name"]==metric) & (df["kernel_operation_mode"]=="full")]
            metrics["memory"] = df_metric["y_value_50"].mean()
            print(MEMORY_PRINT_TEMPLATE.format(
                kernel_name="rope",
                avg_memory=metrics["memory"]
            ))
    return metrics

    # speed_mode = ["forward", "backward"]

    # for mode in speed_mode:
    #     df_speed = df.loc[(df["kernel_operation_mode"]==mode) & (df["metric_name"]=="speed")]
    #     # get the unique x_label
    #     x_labels = df_speed["x_label"].unique()
    #     for x_label in x_labels:
    #         df_x = df_speed.loc[df_speed["x_label"]==x_label]
    #         min_latency = df_x["y_value_50"].min()
    #         max_latency = df_x["y_value_50"].max()
    #         min_x_value = df_x.loc[df_x["y_value_50"]==min_latency, "x_value"].values[0]
    #         max_x_value = df_x.loc[df_x["y_value_50"]==max_latency, "x_value"].values[0]
    #         print(SPEED_PRINT_TEMPLATE.format(
    #         kernel_name="rope",
    #         kernel_operation_mode=mode,
    #         min_latency=f"{min_latency:.2f}",
    #         max_latency=f"{max_latency:.2f}",
    #         x_name=x_label,
    #         min_x_value=min_x_value,
    #         max_x_value=max_x_value,
    #         condition=df_x["extra_benchmark_config_str"].values[0]
    #     ))

    #     print("-"*50)

    # group by kernel_operation_mode, metric_name, x_name
    # grouped = df.groupby(["kernel_operation_mode", "metric_name", "x_name"])
    # # for each group, get the y_value_50 value at x_value is min and max
    # y_value_50 = grouped["y_value_50"].agg(["min", "max"])
    # # print the results for each group
    # # print the corresponding x_value too
    # for (kernel_operation_mode, metric_name, x_name), row in y_value_50.iterrows():
    #     print(f"Kernel Operation Mode: {kernel_operation_mode}, Metric Name: {metric_name}, X Name: {x_name}")
    #     print(f"  Y Value 50 Min: {row['min']:.2f}, X value: {df.loc[(df['kernel_operation_mode'] == kernel_operation_mode) & (df['metric_name'] == metric_name) & (df['x_name'] == x_name) & (df['y_value_50']==row['min']), 'x_value'].values[0]}")
    #     print(f"  Y Value 50 Max: {row['max']:.2f}, X value: {df.loc[(df['kernel_operation_mode'] == kernel_operation_mode) & (df['metric_name'] == metric_name) & (df['x_name'] == x_name) & (df['y_value_50']==row['max']), 'x_value'].values[0]}")
    

if __name__=="__main__":
    csv_file = "all_benchmark_data.csv"
    sample(csv_file)
    parse_benchmark_output("rope_liger_benchmark.csv")