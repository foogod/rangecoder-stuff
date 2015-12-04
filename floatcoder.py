from __future__ import division

import qsmodel
import struct

import sys
def log(msg):
    sys.stderr.write(str(msg) + '\n')

MODEL_BITS = 12
RANGE_TOP = (1 << 32) - 1

def count_zeros(value):
    result = 32
    while value:
        value >>= 1
        result -= 1
    return result

def float_to_int(f):
    i = struct.unpack('I', struct.pack('f', f))[0]
    if i & (1 << 31):
        i ^= 0xffffffff
    else:
        i ^= (1 << 31)
    return i

def int_to_float(i):
    if i & (1 << 31):
        i ^= (1 << 31)
    else:
        i ^= 0xffffffff
    f = struct.unpack('f', struct.pack('I', i))[0]
    return f

class EOFError (Exception):
    pass

class Encoder:
    def __init__(self, outfile, model):
        self.outfile = outfile
        self.model = model
        self.low = 0
        self.range = RANGE_TOP

    def putbyte(self):
        self.outfile.write(chr(self.low >> 24))
        self.low <<= 8
        self.low &= 0xffffffff  # Make sure we stay within a 32-bit value

    def encode_sym(self, sym):
        sym_freq, cum_freq = self.model.getfreq(sym)
        self.range //= self.model.tot_freq
        self.low += cum_freq * self.range
        self.range *= sym_freq
        self.normalize()
        self.model.update(sym)

    # encode a literal binary number into the output stream:
    # 0 <= value < 2^bits <= 2^16
    def encode_value(self, value, bits):
        if bits > 16:
            log("encode_value: value=0x{:x} ({})".format(value, bits))
        while bits > 16:
            self.encode_value(value >> (bits - 16), 16)
            bits -= 16
            value &= (1 << bits) - 1
        log("encode_value: 0x{:04x}/{}".format(value, bits))
        self.range >>= bits
        self.low += value * self.range
        self.normalize()

    def normalize(self):
        while ((self.low ^ (self.low + self.range)) >> 24) == 0:
            self.putbyte()
            self.range <<= 8
        if (self.range >> 16) == 0:
            self.putbyte()
            self.putbyte()
            self.range = -self.low
        self.range &= 0xffffffff  # Make sure we stay within a 32-bit value

    def done(self):
        for i in range(4):
            self.putbyte()


class Decoder:
    def __init__(self, infile, model):
        self.infile = infile
        self.model = model
        self.low = 0
        self.code = 0
        self.range = RANGE_TOP
        for i in xrange(4):
            self.getbyte()

    def getbyte(self):
        c = self.infile.read(1)
        if not c:
            raise EOFError("End of file")
        self.code = (self.code << 8) | ord(c)
        self.low <<= 8
        self.code &= 0xffffffff  # Make sure we stay within a 32-bit value
        self.low &= 0xffffffff  # Make sure we stay within a 32-bit value

    def decode_sym(self):
        self.range //= self.model.tot_freq
        cum_freq = (self.code - self.low) // self.range
        if cum_freq >= self.model.tot_freq:
            raise ValueError("Bad read_byte data")
        sym = self.model.getsym(cum_freq)
        sym_freq, cum_freq = self.model.getfreq(sym)
        self.low += cum_freq * self.range
        self.range *= sym_freq
        self.normalize()
        self.model.update(sym)
        return sym

    def decode_value(self, bits):
        if bits > 16:
            value = 0
            while bits > 16:
                value = (value << 16) | self.decode_value(16)
                bits -= 16
            return (value << bits) | self.decode_value(bits)
        self.range >>= bits
        value = (self.code - self.low) // self.range
        if value >> bits: #FIXME: may be unnecessary
            raise ValueError("Bad compressed data")
        self.low += value * self.range
        self.normalize()
        log("decode_value: 0x{:x}".format(value))
        return value

    def normalize(self):
        while ((self.low ^ (self.low + self.range)) >> 24) == 0:
            self.getbyte()
            self.range <<= 8
        if (self.range >> 16) == 0:
            self.getbyte()
            self.getbyte()
            self.range = -self.low
        self.range &= 0xffffffff  # Make sure we stay within a 32-bit value

    def done(self):
        pass


class Float_Encoder:
    def __init__(self, outfile, predictor):
        self.predictor = predictor
        model = qsmodel.QSModel(64, MODEL_BITS, 2000)
        self.rc = Encoder(outfile, model)

    def encode_float(self, v):
        #FIXME: eventually need to convert to int prior to prediction step (for portability)
        p = float_to_int(self.predictor.next())
        i = float_to_int(v)
        d = (i - p) & 0xffffffff
        log("v={}, i={:08x}, p={:08x}, d={:08x}".format(v, i, p, d))
        self.encode(d)
        self.predictor.update(v)

    def encode(self, value):
        log("Encoding {:08x}".format(value))
        if not (value & (1 << 31)):
            # Positive value
            zeros = count_zeros(value)
            # top bit is zero, so count_zeros will always be between 1 and 32
            sym = zeros - 1
            # result: 0 <= sym <= 31
        else:
            # Negative value
            value = (value & 0xffffffff) ^ 0xffffffff
            zeros = count_zeros(value)
            # top bit is also now zero, so count_zeros will always be between 1 and 32
            sym = 32 | (zeros - 1)
            # result: 32 <= sym <= 63
        log("- sym=0x{:02x}".format(sym))
        self.rc.encode_sym(sym)
        # First bit after zeros is (by definition) always 1, so no need to send it.
        bits = 32 - zeros - 1
        if bits > 0:
            range = 1 << bits
            log("- value=0x{:x} ({})".format(value ^ range, bits))
            self.rc.encode_value(value ^ range, bits)

    def done(self):
        self.rc.done()


class Float_Decoder:
    def __init__(self, infile, predictor):
        self.predictor = predictor
        model = qsmodel.QSModel(64, MODEL_BITS, 2000, compress=False)
        self.rc = Decoder(infile, model)

    def decode_float(self):
        p = float_to_int(self.predictor.next())
        d = self.decode()
        i = (p + d) & 0xffffffff
        v = int_to_float(i)
        log("v={}, i={:08x}, p={:08x}, d={:08x}".format(v, i, p, d))
        self.predictor.update(v)
        return v

    def decode(self):
        sym = self.rc.decode_sym()
        log("- sym=0x{:02x}".format(sym))
        sign = sym >> 5
        zeros = (sym & 0x1f) + 1
        bits = 32 - zeros - 1
        log("- zeros={} bits={}".format(zeros, bits))
        if bits > 0:
            range = 1 << bits
            value = self.rc.decode_value(bits) | range
        elif bits == 0:
            value = 1
        else:
            value = 0
        log("- value=0x{:x} ({})".format(value, bits))
        if sign:
            value ^= 0xffffffff
        return value

    def done(self):
        self.rc.done()


class SuperSimplePredictor:
    def __init__(self, initial_value=0.0):
        self.last_value = initial_value

    def next(self):
        return self.last_value

    def update(self, value):
        self.last_value = value


class SimpleLinearPredictor:
    def __init__(self, initial_value=0.0, interval=1):
        self.initial_value = initial_value
        self.interval = interval
        self.prev = []

    def next(self):
        if not self.prev:
            return self.initial_value
        if len(self.prev) == 1:
            return self.prev[0]
        try:
            slope = (self.prev[-1] - self.prev[-(self.interval + 1)]) / self.interval
        except IndexError:
            slope = (self.prev[-1] - self.prev[0]) / len(self.prev - 1)
        return self.prev[-1] + slope

    def update(self, value):
        self.prev.append(value)
        if len(self.prev) > self.interval + 1:
            del self.prev[0]


class InterleavedLinearPredictor:
    def __init__(self, stride, initial_value=0.0):
        self.stride = stride
        self.initial_value = initial_value
        self.prev = []

    def next(self):
        try:
            return (self.prev[-self.stride] * 2) - self.prev[-self.stride * 2]
        except IndexError:
            # We don't have enough data in prev yet to calculate a trend
            pass
        try:
            return self.prev[-self.stride]
        except IndexError:
            # We don't have even a single previous value for this column yet
            pass
        try:
            return self.prev[-1]
        except IndexError:
            # We don't have any previous values at all
            return self.initial_value

    def update(self, value):
        self.prev.append(value)
        if len(self.prev) > self.stride * 2:
            del self.prev[0]

