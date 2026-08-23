# How this graph maps onto a durable workflow engine

I did not run Temporal, Airflow, Prefect, or n8n in this lab. This note maps Harbor's existing graph onto those engines' vocabulary. If you need a run receipt, this is not it.

## What already exists

The LangGraph wrap has five nodes: `route`, `retrieve`, `draft`, `review`, `emit_receipt`. Review is an interrupt. `emit_receipt` writes once per thread. `receipt_hash` is the stable id.

CLI `ask` is a one-shot trigger. That is the whole production surface today.

## Trigger

- Temporal: start a workflow with the question as input.
- Airflow: a DAG run, from a sensor or from the CLI.
- n8n: a webhook that receives the question.
- Prefect: a flow run.

The trigger is not the interesting part. The interesting part is what happens after a retry.

## Retry

`retrieve` can retry. A timeout on `query_knowledge` is a tool error today. A durable engine would retry that activity.

`draft` with a model can retry. Extractive draft is deterministic, so retry is wasted work but harmless.

`emit_receipt` must not mint a second hash. The graph already refuses that on a finished thread. That is the idempotency key.

In Temporal that key is a workflow id. In Airflow it is a run id plus a skip if the receipt already exists. In n8n it is easier to get wrong, because HTTP nodes retry by default.

## Approval

The review interrupt means a human has to say `approve` or `reject` before a receipt is written.

- Temporal: a signal, or wait-for-update.
- Airflow: a sensor or an approval operator.
- n8n: a Wait node.
- Prefect: pause and resume.

Approve still cannot add tools. That constraint lives in the allowlist, not in the engine. Moving the graph onto Temporal would not make `write_index` legal.

## What I am not claiming

I did not deploy a worker, register a DAG, or store an n8n credential. Prefect is in the same bucket. Zapier and Make are hosted connectors. They are not a substitute for the receipt and allowlist work.

The skill here is knowing where trigger, retry, idempotency, and approval sit. The next time I actually run one of these engines, that run gets its own receipt.
