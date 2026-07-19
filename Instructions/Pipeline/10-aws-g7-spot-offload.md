# Optional AWS G7 Spot offload

Use this only for disposable, interruption-safe T2 experiments. It does not change the local Windows production route.

1. Stage only approved rasters, contract, pinned checkout revision, and dependency manifest in private S3. Never upload secrets, tokens, or entire local virtual environments.
2. Start with `g7.2xlarge`, the smallest G7 shape with one 32 GiB NVIDIA RTX PRO 4500 GPU. Build a Linux AMI/container from the pinned Python 3.11/T2 dependency specification.
3. At launch time, compare `g7.2xlarge` Spot price history and request a fresh placement score across permitted Regions/AZs. Prices and capacity fluctuate; do not record a fixed cheapest Region or rate.
4. Request a one-time EC2 Fleet using `price-capacity-optimized`, not `lowest-price`. It seeks low price while avoiding capacity pools with a higher interruption risk.
5. Checkpoint each completed stage to S3: source hash, settings, selected seed, raw/extracted mesh, validation report, logs, and review renders. React to the two-minute interruption notice by stopping after the current atomic stage and uploading progress.
6. Use private networking and SSM only. Do not expose MCP, ComfyUI, SSH, or the worker. Terminate the instance and remove transient EBS resources after copying reviewed results back.

Query current history before every run:

```powershell
aws ec2 describe-spot-price-history `
  --instance-types g7.2xlarge `
  --product-descriptions 'Linux/UNIX' `
  --start-time (Get-Date).ToUniversalTime().AddDays(-1).ToString('o') `
  --region <region>
```

See AWS documentation for [G7 specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html), [Spot price history](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances-history.html), [placement scores](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-sps-works.html), and [allocation strategy](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-fleet-allocation-strategy.html).