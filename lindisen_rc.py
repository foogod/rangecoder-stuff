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

    def putbyte(self):
        self.write_byte(self.low >> 24)
        self.low <<= 8
        self.low &= 0xffffffff  # Make sure we stay within a 32-bit value

    def normalize(self):
        while ((self.low ^ (self.low + self.range)) >> 24) == 0:
            self.putbyte()
            self.range <<= 8
        if (self.range >> 16) == 0:
            self.putbyte()
            self.putbyte()
            self.range = -self.low
        self.range &= 0xffffffff  # Make sure we stay within a 32-bit value

    def encode_freq(self, sym_freq, cum_freq, tot_freq):
        self.range //= tot_freq
        self.low += cum_freq * self.range
        self.range *= sym_freq
        self.normalize()

    # Encode a literal binary number into the output stream:
    #   0 <= value < 2^bits <= 2^16
    # Note: This is functionally equivalent to
    #   self.encode_freq(1, value, 1 << bits)
    def encode_literal(self, value, bits):
        self.range >>= bits
        self.low += value * self.range
        self.normalize()

    def done(self):
        for i in range(4):
            self.putbyte()


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

    def getbyte(self):
        self.code = (self.code << 8) | self.read_byte()
        self.low <<= 8
        self.code &= 0xffffffff  # Make sure we stay within a 32-bit value
        self.low &= 0xffffffff  # Make sure we stay within a 32-bit value

    def normalize(self):
        while ((self.low ^ (self.low + self.range)) >> 24) == 0:
            self.getbyte()
            self.range <<= 8
        if (self.range >> 16) == 0:
            self.getbyte()
            self.getbyte()
            self.range = -self.low
        self.range &= 0xffffffff  # Make sure we stay within a 32-bit value

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

    def decode_literal(self, bits):
        return self.decode_freq(1 << bits)

    def update_literal(self, value, bits):
        return self.update_freq(1, value, 1 << bits)

    def done(self):
        pass

