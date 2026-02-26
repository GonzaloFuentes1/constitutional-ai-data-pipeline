#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constitutional AI batched pipeline (vLLM version, no llm-swarm).

Main entry point that coordinates all pipeline modules.

- Loads constitution + few-shot examples from a JSON file.
- Runs a 3-stage process: red-teaming -> critique -> revision.
- Uses true vLLM batching to process prompts in parallel.

Outputs:
- Extended constitution copy in exps/<name>_<ts>.json (with `system_chat`).
- Parquet dataset in exps/<name>_<ts>.parquet with CAI training fields.
"""

from config import parse_args


def main():
    """Main function that executes the Constitutional AI pipeline."""
    try:
        # Parse configuration arguments
        args = parse_args()

        # Import heavy pipeline dependencies only after argument parsing.
        # This allows `--help` and config validation to run even in environments
        # where vLLM runtime dependencies are not yet fully available.
        from pipeline import ConstitutionalAIPipeline
        
        # Create and run pipeline
        pipeline = ConstitutionalAIPipeline(args)
        out_json_path, out_parquet_path = pipeline.run()
        
        # Print final summary
        print("\n" + "="*60)
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f" Extended constitution: {out_json_path}")
        print(f" Parquet dataset: {out_parquet_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\n Error running pipeline: {e}")
        raise


if __name__ == "__main__":
    main()