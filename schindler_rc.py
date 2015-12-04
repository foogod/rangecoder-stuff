from __future__ import division

CODE_BITS = 32
SHIFT_BITS = CODE_BITS - 9
EXTRA_BITS = ((CODE_BITS - 2) % 8) + 1
RANGE_TOP = 1 << (CODE_BITS-1)
RANGE_BOTTOM = RANGE_TOP >> 8

class EOFError (Exception):
    pass

class Encoder:
    def __init__(self, outfile, initial_byte=0):
        self.outfile = outfile
        self.low = 0
        self.range = RANGE_TOP
        self.buffer = initial_byte
        self.help = 0

    def write_byte(self, c):
        self.outfile.write(chr(c))

    def normalize(self):
        while self.range <= RANGE_BOTTOM:
            if self.low < (0xff << SHIFT_BITS):
                # No carry possible
                self.write_byte(self.buffer);
                for i in xrange(self.help):
                    self.write_byte(0xff)
                self.help = 0
                self.buffer = (self.low >> SHIFT_BITS) & 0xff
            elif self.low & RANGE_TOP:
                # Carry now
                self.write_byte(self.buffer+1)
                for i in xrange(self.help):
                    self.write_byte(0)
                self.help = 0
                self.buffer = (self.low >> SHIFT_BITS) & 0xff
            else:
                # Potential carry
                self.help += 1
            self.range <<= 8
            self.low = (self.low << 8) & (RANGE_TOP - 1)

    def encode_freq(self, sym_freq, cum_freq, tot_freq):
        self.normalize()
        r = self.range // tot_freq
        tmp = r * cum_freq
        self.low += tmp
        if cum_freq + sym_freq < tot_freq:
            self.range = r * sym_freq
        else:
            self.range -= tmp

    def done(self):
        self.normalize()
        tmp = (self.low >> SHIFT_BITS) + 1
        if tmp > 0xff:
            # Carry
            self.write_byte(self.buffer + 1)
            for i in xrange(self.help):
                self.write_byte(0)
        else:
            self.write_byte(self.buffer)
            for i in xrange(self.help):
                self.write_byte(0xff)
        self.write_byte(tmp & 0xff)
        # These last 3 bytes can be anything.  The decoder reads them but does
        # not actually use them.
        self.write_byte(0)
        self.write_byte(0)
        self.write_byte(0)
        return


class Decoder:
    def __init__(self, infile):
        self.infile = infile
        c = self.read_byte()
        self.buffer = self.read_byte()
        self.low = self.buffer >> (8 - EXTRA_BITS)
        self.range = 1 << EXTRA_BITS

    def read_byte(self):
        c = self.infile.read(1)
        if not c:
            raise EOFError("End of file")
        return ord(c)

    def normalize(self):
        while self.range <= RANGE_BOTTOM:
            self.low = (self.low << 8) | ((self.buffer << EXTRA_BITS) & 0xff)
            self.buffer = self.read_byte()
            self.low |= self.buffer >> (8 - EXTRA_BITS)
            self.range <<= 8

    def decode_freq(self, tot_freq):
        self.normalize()
        self.help = self.range // tot_freq
        cum_freq = self.low // self.help
        if cum_freq >= tot_freq:
            return tot_freq - 1
        else:
            return cum_freq

    def update(self, sym_freq, cum_freq, tot_freq):
        tmp = self.help * cum_freq
        self.low -= tmp
        if cum_freq + sym_freq < tot_freq:
            self.range = self.help * sym_freq
        else:
            self.range -= tmp

    def done(self):
        self.normalize()

