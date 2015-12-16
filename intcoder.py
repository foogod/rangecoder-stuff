from __future__ import division

import qsmodel

import sys

verbose = False

def log(msg):
    if verbose:
        sys.stderr.write(str(msg) + '\n')

MODEL_BITS = 16
RANGE_TOP = (1 << 32) - 1

def count_zeros(value):
    result = 32
    while value:
        value >>= 1
        result -= 1
    return result

def count_zeros64(value):
    result = 64
    while value:
        value >>= 1
        result -= 1
    return result

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
        while bits > 16:
            self.encode_value(value >> (bits - 16), 16)
            bits -= 16
            value &= (1 << bits) - 1
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


class NoOpEncoder:
    def __init__(self, outfile, sym_size):
        self.outfile = outfile
        self.sym_size = sym_size
        self.buffer = 0
        self.bits = 0

    def putbyte(self):
        c = self.buffer >> (self.bits - 8)
        log(" -- write: 0x{:02x}".format(c))
        self.outfile.write(chr(c))
        self.bits -= 8
        self.buffer &= (1 << self.bits) - 1

    def encode_sym(self, sym):
        self.encode_value(sym, self.sym_size)

    def encode_value(self, value, bits):
        self.buffer = (self.buffer << bits) | value
        self.bits += bits
        while self.bits >= 8:
            self.putbyte()

    def done(self):
        self.buffer <<= (8 - self.bits)
        self.bits = 8
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


# IntZ -- Code initial zeros + residual.

class IntZ_Encoder:
    def __init__(self, outfile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(63, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.count = 0

    def encode_int(self, v):
        p = self.predictor.next()
        d = (v - p)
        log("E:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.encode(d)
        self.predictor.update(v)
        self.count += 1

    def encode(self, value):
        if value < 0:
            sign = 1
            value = -value
        else:
            sign = 0
        zeros = count_zeros(value) - 1
        sym = (sign << 5) | zeros
        bits = 30 - zeros
        if bits > 0:
            value &= (1 << bits) - 1
            log("- sym={:02x} value={:08x} bits={}".format(sym, value, bits))
            self.rc.encode_sym(sym)
            self.rc.encode_value(value, bits)
        else:
            log("- sym={:02x} bits={}".format(sym, bits))
            self.rc.encode_sym(sym)

    def done(self):
        self.rc.done()


class IntZ_Decoder:
    def __init__(self, infile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(63, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.count = 0

    def decode_int(self):
        p = self.predictor.next()
        d = self.decode()
        v = (p + d) & 0xffffffff
        log("D:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.predictor.update(v)
        self.count += 1
        return v

    def decode(self):
        sym = self.rc.decode_sym()
        zeros = sym & 0x1f
        bits = 30 - zeros
        if bits > 0:
            value = self.rc.decode_value(bits)
            log("- sym={:02x} value={:08x} bits={}".format(sym, value, bits))
            value |= (1 << bits)
        elif bits == 0:
            value = 1
        else:
            value = 0
        if sym & 0x20:
            value = -value
        return value

    def done(self):
        self.rc.done()


# IntSV -- Code small values as symbols, others as full 32-bit literal

class IntSV_Encoder:
    def __init__(self, outfile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(64, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.count = 0

    def encode_int(self, v):
        p = self.predictor.next()
        d = (v - p)
        log("E:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.encode(d)
        self.predictor.update(v)
        self.count += 1

    def encode(self, value):
        if value < 32 and value > -32:
            log("- sym={:02x}".format(value + 31))
            self.rc.encode_sym(value + 31)
        else:
            log("- sym={:02x} value={:08x}".format(63, value))
            self.rc.encode_sym(63)
            self.rc.encode_value(value & 0xffffffff, 32)

    def done(self):
        self.rc.done()


class IntSV_Decoder:
    def __init__(self, infile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(64, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.count = 0

    def decode_int(self):
        p = self.predictor.next()
        d = self.decode()
        v = (p + d) & 0xffffffff
        log("D:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.predictor.update(v)
        self.count += 1
        return v

    def decode(self):
        sym = self.rc.decode_sym()
        if sym < 63:
            log("- sym={:02x}".format(sym))
            return sym - 31
        else:
            value = self.rc.decode_value(32)
            log("- sym={:02x} value={:08x}".format(sym, value))
            return value

    def done(self):
        self.rc.done()


# IntSVZ -- Code small values as symbols.  Others as zeros + literal

class IntSVZ_Encoder:
    def __init__(self, outfile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(128, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.count = 0

    def encode_int(self, v):
        p = self.predictor.next()
        d = (v - p)
        log("E:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.encode(d)
        self.predictor.update(v)
        self.count += 1

    def encode(self, value):
        if value < 32 and value > -33:
            log("- sym={:02x}".format(value + 32))
            self.rc.encode_sym(value + 32)
        else:
            # Value outside usual range.  Code it as leading-zeros + literal
            if value < 0:
                sign = 1
                value = -value
            else:
                sign = 0
            zeros = count_zeros(value)
            sym = 0x40 | (sign << 5) | zeros
            bits = 31 - zeros
            value &= (1 << bits) - 1
            log("- sym={:02x} value={:08x} bits={}".format(sym, value, bits))
            self.rc.encode_sym(sym)
            self.rc.encode_value(value, bits)

    def done(self):
        self.rc.done()


class IntSVZ_Decoder:
    def __init__(self, infile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(128, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.count = 0

    def decode_int(self):
        p = self.predictor.next()
        d = self.decode()
        v = (p + d) & 0xffffffff
        log("D:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.predictor.update(v)
        self.count += 1
        return v

    def decode(self):
        sym = self.rc.decode_sym()
        if sym < 0x40:
            log("- sym={:02x}".format(sym))
            return sym - 32
        else:
            zeros = sym & 0x1f
            bits = 31 - zeros
            value = self.rc.decode_value(bits)
            log("- sym={:02x} value={:08x} bits={}".format(sym, value, 32 - zeros))
            value |= (1 << bits)
            if sym & 0x20:
                value = -value
            return value

    def done(self):
        self.rc.done()


class IntZ64_Encoder:
    def __init__(self, outfile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(127, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.count = 0

    def encode_int(self, v):
        p = self.predictor.next()
        d = (v - p)
        log("E:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.encode(d)
        self.predictor.update(v)
        self.count += 1

    def encode(self, value):
        value &= (1 << 64) - 1
        if value & (1 << 63):
            sign = 1
            value = -value & ((1 << 64) - 1)
        else:
            sign = 0
        zeros = count_zeros64(value) - 1
        sym = (sign << 6) | zeros
        bits = 62 - zeros
        if bits > 0:
            value &= (1 << bits) - 1
            log("- sym={:02x} value={:08x} bits={}".format(sym, value, bits))
            self.rc.encode_sym(sym)
            self.rc.encode_value(value, bits)
        else:
            log("- sym={:02x} bits={}".format(sym, bits))
            self.rc.encode_sym(sym)

    def done(self):
        self.rc.done()


class IntZ64_Decoder:
    def __init__(self, infile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(127, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.count = 0

    def decode_int(self):
        p = self.predictor.next()
        d = self.decode()
        v = (p + d) & ((1 << 64) - 1)
        log("D:{}: v={:08x}, p={:08x}, d={:08x}".format(self.count, v, p, d))
        self.predictor.update(v)
        self.count += 1
        return v

    def decode(self):
        sym = self.rc.decode_sym()
        zeros = sym & 0x3f
        bits = 62 - zeros
        if bits > 0:
            value = self.rc.decode_value(bits)
            log("- sym={:02x} value={:08x} bits={}".format(sym, value, bits))
            value |= (1 << bits)
        elif bits == 0:
            value = 1
        else:
            value = 0
        if sym & 0x40:
            value = -value
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


class InterleavedSimplePredictor:
    def __init__(self, stride, initial_value=0):
        self.stride = stride
        self.initial_value = initial_value
        self.prev = []

    def next(self):
        try:
            return self.prev[-self.stride]
        except IndexError:
            # We don't have a previous value for this column yet
            pass
        try:
            return self.prev[-1]
        except IndexError:
            # We don't have any previous values at all
            return self.initial_value

    def update(self, value):
        self.prev.append(value)
        if len(self.prev) > self.stride:
            del self.prev[0]


class InterleavedLinearPredictor:
    def __init__(self, stride, initial_value=0):
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


class MultiPredictor:
    def __init__(self, *predictors):
        self.predictors = predictors
        self.nexts = None
        self.freqs = [[0] * len(predictors) for p in predictors]
        self.prev_best = 0
        self.rescale_limit = len(predictors) ** 2 * 500
        self.count = 0

    def next(self):
        self.nexts = [p.next() for p in self.predictors]
        freqs = self.freqs[self.prev_best]
        cur_max = 0
        cur_i = 0
        for i in xrange(len(freqs)):
            if freqs[i] > cur_max:
                cur_max = freqs[i]
                cur_i = i
        log("P: nexts={} chosen={}".format(self.nexts, cur_i))
        return self.nexts[cur_i]

    def update(self, v):
        for p in self.predictors:
            p.update(v)
        if self.count >= self.rescale_limit:
            self.rescale()
            self.count = 0
        self.count += 1
        cur_min = 1 << 32 #FIXME
        cur_i = 0
        for i in xrange(len(self.nexts)):
            delta = abs(v - self.nexts[i])
            if delta < cur_min:
                cur_min = delta
                cur_i = i
        self.freqs[self.prev_best][cur_i] += 1
        log("P: actual_best={}".format(cur_i))
        self.prev_best = cur_i

    def rescale(self):
        freqs = self.freqs
        log("P: Rescaling:")
        for i in xrange(len(freqs)):
            log(" - {}: {}".format(i, freqs[i]))
            for j in xrange(len(freqs[i])):
                freqs[i][j] >>= 1


class Multi2Predictor:
    def __init__(self, *predictors):
        self.predictors = predictors
        self.nexts = None
        self.freqs = [[[0] * len(predictors) for p in predictors] for p in predictors]
        self.prev_best = [0, 0]
        self.rescale_limit = len(predictors) ** 3 * 500
        self.count = 0

    def next(self):
        self.nexts = [p.next() for p in self.predictors]
        freqs = self.freqs[self.prev_best[0]][self.prev_best[1]]
        cur_max = 0
        cur_i = 0
        for i in xrange(len(freqs)):
            if freqs[i] > cur_max:
                cur_max = freqs[i]
                cur_i = i
        log("P: nexts={} chosen={}".format(self.nexts, cur_i))
        self.chosen = cur_i
        return self.nexts[cur_i]

    def update(self, v):
        for p in self.predictors:
            p.update(v)
        if self.count >= self.rescale_limit:
            self.rescale()
            self.count = 0
        self.count += 1
        cur_min = 1 << 32 #FIXME
        cur_i = 0
        for i in xrange(len(self.nexts)):
            delta = abs(v - self.nexts[i])
            if delta < cur_min:
                cur_min = delta
                cur_i = i
        self.freqs[self.prev_best[0]][self.prev_best[1]][cur_i] += 1
        if cur_i == self.chosen:
            log("P: chosen={} best={} (success!)".format(self.chosen, cur_i))
        else:
            log("P: chosen={} best={} (oops)".format(self.chosen, cur_i))
        self.prev_best = [self.prev_best[1], cur_i]

    def rescale(self):
        freqs = self.freqs
        log("P: Rescaling:")
        for i in xrange(len(freqs)):
            for j in xrange(len(freqs[i])):
                log(" - [{}, {}]: {}".format(i, j, freqs[i][j]))
                for k in xrange(len(freqs[i][j])):
                    freqs[i][j][k] >>= 1


Int_Encoder = IntZ_Encoder
Int_Decoder = IntZ_Decoder

Int64_Encoder = IntZ64_Encoder
Int64_Decoder = IntZ64_Decoder
