import matplotlib

# Force a non-interactive backend before anything under stmr can import pyplot. Training
# runs generate matplotlib figures (tensorboard debug images, PDF summaries) from threads
# other than the main one (e.g. the CUDA autograd worker running loss.backward()); the
# default interactive backend (TkAgg, if Tk is installed) is not thread-safe and crashes
# with "main thread is not in main loop" / Tcl_AsyncDelete aborts when a figure created on
# one thread is garbage-collected from another. Nothing in this codebase calls plt.show().
matplotlib.use("Agg")
