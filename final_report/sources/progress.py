#progress bar package, from:https://gist.github.com/vladignatyev/06860ec2040cb497f0f3, free-license use
import sys
def bar(count, total, suffix=''):
	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))

	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)

	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
	sys.stdout.flush()  # As suggested by Rom Ruben
	pass
