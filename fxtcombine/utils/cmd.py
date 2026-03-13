#!/usr/bin/env python3
"""

"""
import subprocess
import os

from fxtcombine.utils.logger import emit


def _emit_command_output(
		output,
		logger=None,
		level="error",
		max_lines=40,
	):
	"""Emit the tail of captured command output through the logger.

	Parameters
	----------
	output : str | None
		Captured combined stdout/stderr text.
	logger : logging.Logger | None, optional
		Optional logger used for output messages.
	level : str, optional
		Log level used when emitting the output.
	max_lines : int, optional
		Maximum number of trailing lines shown through the logger.

	Returns
	-------
	None
	"""
	if not output:
		return
	lines = output.rstrip().splitlines()
	if not lines:
		return
	if len(lines) > max_lines:
		emit(
			logger,
			level,
			f"Command output truncated to the last {max_lines} lines; see the step log for the full output.",
		)
		lines = lines[-max_lines:]
	for line in lines:
		emit(logger, level, line)


def run_cmd(
		cmd_str,
		logger=None,logname="./run.log"
	):
	"""Run one shell command and raise on failure.

	Parameters
	----------
	cmd_str : str
		Command string executed through the shell.
	logger : logging.Logger | None, optional
		Optional logger used for command and error messages.
	logname : str, optional
		Path of the file used to store combined stdout and stderr.

	Returns
	-------
	subprocess.CompletedProcess
		Completed-process object returned by :func:`subprocess.run`.

	Raises
	------
	subprocess.CalledProcessError
		Raised when the command exits with a non-zero status.
	"""
	emit(logger, "info", cmd_str)
	result = subprocess.run(
		cmd_str,
		shell=True,
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
	)
	if logger is not None:
		with open(logname, "w") as log_file:
			log_file.write(result.stdout or "")
	if result.returncode != 0:
		emit(logger, "error", f"Command failed with exit code {result.returncode}: {cmd_str}")
		_emit_command_output(result.stdout, logger=logger, level="error")
		raise subprocess.CalledProcessError(
			result.returncode,
			cmd_str,
			output=result.stdout,
		)
	return result


def remove_xselect_tmp_files():
	"""Remove temporary xselect working files from the current directory.

	Returns
	-------
	None
	"""
	tmp_fname_lst = [
		"EP_ascii_out.xsl","EP_display.def","EP_files.tmp","EP_hist.xsl","EP_obscat.tmp","EP_obslist.def","EP_read_cat.xsl","EP_region.xsl","EP_xsel.run"
	]
	for tmp_fname in tmp_fname_lst:
		if os.path.exists(tmp_fname):
			os.remove(tmp_fname)
