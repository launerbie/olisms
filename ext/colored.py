import unittest
import time
 
_COLOR = {'green': "\x1b[32;01m",
          'red': "\x1b[31;01m",
          'reset': "\x1b[0m"
          }
 
def red_str(text):
    """Return red text."""
    global _COLOR
    return _COLOR['red'] + text + _COLOR['reset']
 
def green_str(text):
    """Return green text."""
    global _COLOR
    return _COLOR['green'] + text + _COLOR['reset']
 
 
class _ColoredTextTestResult(unittest._TextTestResult):
    """Colored version."""
    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        if self.showAll:
            self.stream.writeln(green_str("Ok"))
        elif self.dots:
            self.stream.write(green_str('.'))
 
    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        if self.showAll:
            self.stream.writeln(red_str("ERROR"))
        elif self.dots:
            self.stream.write(red_str('E'))
 
    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        if self.showAll:
            self.stream.writeln(red_str("FAIL"))
        elif self.dots:
            self.stream.write(red_str('F'))
 
    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln("%s: %s" % (red_str(flavour),
                                            self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)
 
 
class ColoredTextTestRunner(unittest.TextTestRunner):
    """Override to be color powered."""
    def _makeResult(self):
        return _ColoredTextTestResult(self.stream,
                                      self.descriptions, self.verbosity)
 
    def run(self, test):
        "Run the given test case or test suite."
        result = self._makeResult()
        startTime = time.time()
        test(result)
        stopTime = time.time()
        timeTaken = float(stopTime - startTime)
        result.printErrors()
        self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()
        if not result.wasSuccessful():
            self.stream.write(red_str("FAILED") + " (")
            failed, errored = map(len, (result.failures, result.errors))
            if failed:
                self.stream.write("failures=%d" % failed)
            if errored:
                if failed: self.stream.write(", ")
                self.stream.write("errors=%d" % errored)
            self.stream.writeln(")")
        else:
            self.stream.writeln(green_str("OK"))
        return result
