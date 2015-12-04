# The following is an implementation of the "quasistatic probability model"
# from Michael Schindler's range coder.  It can theoretically be used with any
# variant of range coding, however, and thus is implemented as a separate
# module.

from __future__ import division

TBLSHIFT = 7

class InternalError (RuntimeError):
    pass


class QSModel:
    def __init__(self, n, lg_totf, rescale, init=None, compress=True):
        self.n = n
        self.tot_freq = 1 << lg_totf
        self.targetrescale = rescale
        self.searchshift = lg_totf - TBLSHIFT
        if self.searchshift < 0:
            self.searchshift = 0
        self.cf = [0] * (n+1)
        self.newf = [0] * (n+1)
        self.cf[n] = 1 << lg_totf
        if compress:
            self.search = None
        else:
            self.search = [0] * ((1 << TBLSHIFT) + 1)
            self.search[1 << TBLSHIFT] = n - 1
        self.incr = 0
        self.left = 0
        self.reset(init)

    def reset(self, init=None):
        self.rescale = (self.n >> 4) | 2
        self.nextleft = 0
        if not init:
            initval = self.cf[self.n] // self.n
            end = self.cf[self.n] % self.n
            for i in xrange(end):
                self.newf[i] = initval + 1
            for i in xrange(end, self.n):
                self.newf[i] = initval
        else:
            for i in xrange(self.n):
                self.newf[i] = init[i]
        self.do_rescale()

    def do_rescale(self):
        if self.nextleft:
            self.incr += 1
            self.left = self.nextleft
            self.nextleft = 0
            return
        if self.rescale < self.targetrescale:
            self.rescale <<= 1
            if self.rescale > self.targetrescale:
                self.rescale = self.targetrescale
        cf = missing = self.cf[self.n]
        for i in xrange(self.n - 1, 0, -1):
            tmp = self.newf[i]
            cf -= tmp
            self.cf[i] = cf
            tmp = (tmp >> 1) | 1
            missing -= tmp
            self.newf[i] = tmp
        if cf != self.newf[0]:
            raise InternalError("BUG: rescaling left {} total frequency".format(cf))
        self.newf[0] = (self.newf[0] >> 1) | 1
        missing -= self.newf[0]
        self.incr = missing // self.rescale
        self.nextleft = missing % self.rescale
        self.left = self.rescale - self.nextleft
        if self.search:
            i = self.n
            while i:
                end = (self.cf[i] - 1) >> self.searchshift
                i -= 1
                start = self.cf[i] >> self.searchshift
                while start <= end:
                    self.search[start] = i
                    start += 1

    def getfreq(self, sym):
        lt_f = self.cf[sym]
        sy_f = self.cf[sym+1] - lt_f
        return (sy_f, lt_f)

    def getsym(self, lt_f):
        search_i = lt_f >> self.searchshift
        lo = self.search[search_i]
        hi = self.search[search_i + 1] + 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if lt_f < self.cf[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    def update(self, sym):
        if self.left <= 0:
            self.do_rescale()
        self.left -= 1
        self.newf[sym] += self.incr


