"""Visualisation tools for inspecting learned motion.

deform.py: propagate a frame through the learned velocity flows and compare to GT.
tb_gif.py: turn images logged to TensorBoard into per-tag training-evolution GIFs.
post_run.py: run both of the above for a just-finished pipeline run (see stmr.pipeline.run),
into <log_path>/gifs and <log_path>/figs.
"""
