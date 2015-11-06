__author__ = 'Kevin'

def print_summary_sentence_safe(summary):

    for s in summary:
        try:
            print(s)
        except:
            print('Error, sentence cannot be printed')

class Lookahead:
    def __init__(self, iter):
        self.iter = iter
        self.buffer = []
        self.curr=None

    def __iter__(self):
        return self

    def next(self):
        self.curr=self.buffer.pop(0) if len(self.buffer)>0 else next(self.iter)

        return self.curr

    def current(self):
        try:
            self.curr=self.curr if self.curr is not None else self.next()

        except StopIteration:
            self.curr=None

        return self.curr

    def lookahead(self, n=1):
        """Return an item n entries ahead in the iteration."""
        while n >= len(self.buffer):
            try:
                self.buffer.append(next(self.iter))
            except StopIteration:
                return None
        return self.buffer[n]

