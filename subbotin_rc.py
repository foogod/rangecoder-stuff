from __future__ import division

TOP = 1 << 24
BOT = 1 << 16
RANGE_TOP = (1 << 32) - 1

class EOFError (Exception):
    pass

class Encoder:
    def __init__(self, outfile):
        self.outfile = outfile
        self.low = 0
        self.range = RANGE_TOP

    def write_byte(self, c):
        self.outfile.write(chr(c))

    def normalize(self):
        while True:
            if (self.low ^ (self.low + self.range)) < TOP:
                pass
            elif self.range < BOT:
                self.range = -self.low & (BOT - 1)
            else:
                break
            self.write_byte(self.low >> 24)
            self.range <<= 8
            self.low <<= 8
            self.range &= 0xffffffff  # Make sure we stay within a 32-bit value
            self.low &= 0xffffffff  # Make sure we stay within a 32-bit value

    def encode_freq(self, sym_freq, cum_freq, tot_freq):
        self.range //= tot_freq
        self.low += cum_freq * self.range
        self.range *= sym_freq
        self.normalize()

    def done(self):
        for i in range(4):
            self.write_byte(self.low >> 24)
            self.low <<= 8
            self.low &= 0xffffffff  # Make sure we stay within a 32-bit value


class Decoder:
    def __init__(self, infile):
        self.infile = infile
        self.low = 0
        self.code = 0
        self.range = RANGE_TOP
        for i in xrange(4):
            self.code = (self.code << 8) | self.read_byte()

    def read_byte(self):
        c = self.infile.read(1)
        if not c:
            raise EOFError("End of file")
        return ord(c)

    def normalize(self):
        while True:
            if (self.low ^ (self.low + self.range)) < TOP:
                pass
            elif self.range < BOT:
                self.range = -self.low & (BOT - 1)
            else:
                break
            self.code = (self.code << 8) | self.read_byte()
            self.range <<= 8
            self.low <<= 8
            self.code &= 0xffffffff  # Make sure we stay within a 32-bit value
            self.range &= 0xffffffff  # Make sure we stay within a 32-bit value
            self.low &= 0xffffffff  # Make sure we stay within a 32-bit value

    def decode_freq(self, tot_freq):
        self.range //= tot_freq
        cum_freq = (self.code - self.low) // self.range
        if cum_freq >= tot_freq:
            raise ValueError("Bad read_byte data")
        return cum_freq

    def update(self, sym_freq, cum_freq, tot_freq):
        self.low += cum_freq * self.range
        self.range *= sym_freq
        self.normalize()

    def done(self):
        pass

