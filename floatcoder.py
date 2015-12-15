from __future__ import division

import qsmodel
import struct

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

def float_to_int(f):
    return struct.unpack('I', struct.pack('f', f))[0]

def int_to_float(i):
    return struct.unpack('f', struct.pack('I', i))[0]

def flint_trans(i):
    if i & (1 << 31):
        i ^= 0xffffffff
    else:
        i ^= (1 << 31)
    return i

def flint_untrans(i):
    if i & (1 << 31):
        i ^= (1 << 31)
    else:
        i ^= 0xffffffff
    return i

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


class Float_Encoder:
    def __init__(self, outfile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(64, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.count = 0

    def encode_float(self, v):
        i = float_to_int(v)
        #FIXME: eventually need to convert to int prior to prediction step (for portability)
        p = flint_trans(float_to_int(self.predictor.next()))
        i = flint_trans(i)
        d = (i - p) & 0xffffffff
        log("E:{}: v={}, i={:08x}, p={:08x}, d={:08x}".format(self.count, v, i, p, d))
        self.encode(d)
        self.predictor.update(int_to_float(flint_untrans(i)))
        self.count += 1

    def encode(self, value):
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
    def __init__(self, infile, predictor, coder=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(64, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.count = 0

    def decode_float(self):
        p = flint_trans(float_to_int(self.predictor.next()))
        d = self.decode()
        i = (p + d) & 0xffffffff
        v = int_to_float(flint_untrans(i))
        log("D:{}: v={}, i={:08x}, p={:08x}, d={:08x}".format(self.count, v, i, p, d))
        self.predictor.update(v)
        self.count += 1
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


class Float_Encoder_SPR:
    def __init__(self, outfile, predictor, coder=None, spr=0):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(64, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.spr = spr
        self.spr_mask = 0xffffffff ^ ((1 << spr) - 1)
        self.count = 0

    def encode_float(self, v):
        #FIXME: eventually need to convert to int prior to prediction step (for portability)
        p = float_to_int(self.predictor.next()) & self.spr_mask
        i = float_to_int(v) & self.spr_mask
        d = (i - p) & 0xffffffff
        log("E:{}: v={}, i={:08x}, p={:08x}, d={:08x}".format(self.count, v, i, p, d))
        self.encode(d)
        self.predictor.update(int_to_float(i))
        self.count += 1

    def encode(self, value):
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
        bits = 32 - zeros - 1 - self.spr
        if bits > 0:
            value >>= self.spr
            range = 1 << bits
            log("- value=0x{:x} ({})".format(value ^ range, bits))
            self.rc.encode_value(value ^ range, bits)

    def done(self):
        self.rc.done()


class Float_Decoder_SPR:
    def __init__(self, infile, predictor, coder=None, spr=0):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(64, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.spr = spr
        self.spr_mask = 0xffffffff ^ ((1 << spr) - 1)
        self.count = 0

    def decode_float(self):
        p = float_to_int(self.predictor.next()) & self.spr_mask
        d = self.decode()
        i = (p + d) & 0xffffffff
        v = int_to_float(i)
        log("D:{}: v={}, i={:08x}, p={:08x}, d={:08x}".format(self.count, v, i, p, d))
        self.predictor.update(v)
        self.count += 1
        return v

    def decode(self):
        sym = self.rc.decode_sym()
        log("- sym=0x{:02x}".format(sym))
        sign = sym >> 5
        zeros = (sym & 0x1f) + 1
        bits = 32 - zeros - 1 - self.spr
        log("- zeros={} bits={}".format(zeros, bits))
        if bits > 0:
            range = 1 << bits
            value = self.rc.decode_value(bits) | range
        elif bits == 0:
            value = 1
        else:
            value = 0
        value <<= self.spr
        log("- value=0x{:x} ({})".format(value, bits))
        if sign:
            value ^= self.spr_mask
        return value

    def done(self):
        self.rc.done()


class Float_Encoder_DPR:
    def __init__(self, outfile, predictor, coder=None, dpr=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(65, MODEL_BITS, 2000)
            self.rc = Encoder(outfile, model)
        self.dpr = dpr
        self.count = 0

    def encode_float(self, v):
        i = float_to_int(v)
        dpr_exp = ((i >> 23) & 0xff) - 0x7f
        log("-- dpr_exp={}".format(dpr_exp))
        if dpr_exp > 0x7f:
            # Extranormal value (infinity, NaN).. don't mess with these.
            dpr_bits = 23
        else:
            dpr_bits = dpr_exp - self.dpr
            if dpr_bits < 0:
                # Value is below our minimum threshold, just make it zero
                dpr_bits = 0
                i = 0
            elif dpr_bits > 23:
                # No rounding necessary
                dpr_bits = 23
        trim_bits = 23 - dpr_bits
        trim_mask = (1 << 32) - (1 << trim_bits)

        #FIXME: eventually need to convert to int prior to prediction step (for portability)
        p = flint_trans(float_to_int(self.predictor.next())) & trim_mask
        i = flint_trans(i) & trim_mask
        p_exp = p & 0xff800000
        p_mant = p & 0x007fffff
        i_exp = i & 0xff800000
        i_mant = i & 0x007fffff
        if i > p:
            sign = 0
            d_exp = (i_exp - p_exp) & 0xff800000
            d_mant = (i_mant - p_mant) & 0x007fffff
        else:
            sign = 1
            d_exp = (p_exp - i_exp) & 0xff800000
            d_mant = (p_mant - i_mant) & 0x007fffff
        d = d_exp | d_mant
        #d = (i - p) & 0xffffffff
        log("E:{}: v={}, i={:08x}/{:08x}:{:08x}, p={:08x}/{:08x}:{:08x}, d={:08x}/{:08x}:{:08x}".format(self.count, v, i, i_exp, i_mant, p, p_exp, p_mant, d, d_exp, d_mant))
        self.encode(sign, d, trim_bits)
        pu = int_to_float(flint_untrans(i) & trim_mask)
        log("! updating predictor with: {}".format(pu))
        self.predictor.update(pu)
        self.count += 1

    def encode(self, sign, value, trim_bits):
        # top bit is always zero, so count_zeros will always be between 1 and 32
        zeros = count_zeros(value)
        if value == 0:
            sym = 64
        elif not sign:
            # Positive value
            sym = zeros
            # result: 0 <= sym <= 31
        else:
            # Negative value
            sym = 32 | zeros
            # result: 32 <= sym <= 63
        log("- sym=0x{:02x}".format(sym))
        self.rc.encode_sym(sym)
        # First bit after zeros is (by definition) always 1, so no need to send it.
        bits = 32 - zeros - 1
        if bits > 23:
            exp_bits = bits - 23
            mant_bits = 23
            range = 1 << bits
            bitval = value ^ range
        elif bits > 0:
            exp_bits = 0
            mant_bits = bits
            range = 1 << bits
            bitval = value ^ range
        else:
            exp_bits = 0
            mant_bits = 0
            bitval = 0

        mant_bits = max(mant_bits - trim_bits, 0)

        log("- zeros={} bits={} exp_bits={} mant_bits={} trim_bits={}".format(zeros, bits, exp_bits, mant_bits, trim_bits))
        if exp_bits:
            log("- encoding exp_bits: {:02x}/{}".format(bitval >> 23, exp_bits))
            self.rc.encode_value(bitval >> 23, exp_bits)
        if mant_bits:
            mantval = (bitval & ((1 << 23) - 1)) >> trim_bits
            log("- encoding mant_bits: {:x}/{}".format(mantval, mant_bits))
            self.rc.encode_value(mantval, mant_bits)

    def done(self):
        self.rc.done()


class Float_Decoder_DPR:
    def __init__(self, infile, predictor, coder=None, dpr=None):
        self.predictor = predictor
        if coder:
            self.rc = coder
        else:
            model = qsmodel.QSModel(65, MODEL_BITS, 2000, compress=False)
            self.rc = Decoder(infile, model)
        self.dpr = dpr
        self.count = 0

    def decode_float(self):
        p = flint_trans(float_to_int(self.predictor.next()))
        p_exp = p & 0xff800000
        p_mant = p & 0x007fffff
        sign, d, trim_bits = self.decode(p_exp)
        trim_mask = (1 << 32) - (1 << trim_bits)
        p_mant &= trim_mask
        d_exp = d & 0xff800000
        d_mant = d & 0x007fffff
        if sign:
            i_exp = (p_exp - d_exp) & 0xff800000
            i_mant = (p_mant - d_mant) & 0x007fffff
        else:
            i_exp = (p_exp + d_exp) & 0xff800000
            i_mant = (p_mant + d_mant) & 0x007fffff
        i = i_exp | i_mant
        #i = (p + d) & 0xffffffff
        v = int_to_float(flint_untrans(i) & trim_mask)
        log("D:{}: v={}, i={:08x}/{:08x}:{:08x}, p={:08x}/{:08x}:{:08x}, d={:08x}/{:08x}:{:08x}".format(self.count, v, i, i_exp, i_mant, p, p_exp, p_mant, d, d_exp, d_mant))
        log("! updating predictor with: {}".format(v))
        self.predictor.update(v)
        self.count += 1
        return v

    def decode(self, p_exp):
        sym = self.rc.decode_sym()
        log("- sym=0x{:02x}".format(sym))
        if sym == 64:
            sign = 0
            zeros = 32
        else:
            sign = sym >> 5
            zeros = (sym & 0x1f)
        if zeros <= 8:
            # We need to decode some literal bits of the exponent first
            exp_bits = 8 - zeros
            range = 1 << exp_bits
            expval = self.rc.decode_value(exp_bits)
            log("- decoding exp_bits: {:02x}/{}".format(expval, exp_bits))
            expval |= range
        else:
            exp_bits = 0
            expval = 0

        d_exp = expval << 23
        if sign:
            i_exp = (p_exp - d_exp) & 0xff800000
        else:
            i_exp = (p_exp + d_exp) & 0xff800000
        dpr_exp = ((flint_untrans(i_exp) >> 23) & 0xff) - 0x7f
        log("-- dpr_exp={}".format(dpr_exp))
        if dpr_exp > 0x7f:
            # Extranormal value (infinity, NaN).. don't mess with these.
            dpr_bits = 23
        else:
            dpr_bits = dpr_exp - self.dpr
            if dpr_bits < 0:
                dpr_bits = 0
                i = 0
            elif dpr_bits > 23:
                dpr_bits = 23
        trim_bits = 23 - dpr_bits

        bits = 32 - zeros - 1
        mant_bits = max(bits - exp_bits - trim_bits, 0)
        log("- zeros={} bits={} exp_bits={} mant_bits={} trim_bits={}".format(zeros, bits, exp_bits, mant_bits, trim_bits))
        if mant_bits:
            value = self.rc.decode_value(mant_bits)
            log("- decoding mant_bits: {:x}/{}".format(value, mant_bits))
            if not exp_bits:
                range = 1 << mant_bits
                value |= range
        elif bits >= 0 and not exp_bits:
            value = 1
        else:
            value = 0
        log("- expval={:02x} valbits={:x}".format(expval, value))
        value <<= trim_bits
        value |= (expval << 23)
        log("- value=0x{:x} ({})".format(value, bits))
        return sign, value, trim_bits

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


class InterleavedTrapezoidPredictor:
    def __init__(self, stride, initial_value=0.0):
        self.stride = stride
        self.initial_value = initial_value
        self.prev = []

    def next(self):
        try:
            return self.prev[-self.stride] + self.prev[-self.stride * 2] - self.prev[-self.stride * 3]
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
        if len(self.prev) > self.stride * 3:
            del self.prev[0]


class InterleavedOrder2Predictor:
    def __init__(self, stride, initial_value=0.0, weight=1.0):
        self.stride = stride
        self.initial_value = initial_value
        self.weight = weight
        self.prev = []

    def next(self):
        try:
            delta = self.prev[-self.stride] - self.prev[-self.stride * 2]
            try:
                prev_delta = self.prev[-self.stride * 2] - self.prev[-self.stride * 3]
            except IndexError:
                # We have one stride's worth, but not 2.  Settle for simple
                # linear result.
                prev_delta = delta
            delta = delta + ((delta - prev_delta) * self.weight)
            return self.prev[-self.stride] + delta
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
        if len(self.prev) > self.stride * 3:
            del self.prev[0]


class InterleavedAdjustedPredictor:
    def __init__(self, stride, initial_value=0.0):
        self.stride = stride
        self.initial_value = initial_value
        self.prev = []
        self.guessed_delta = 0
        self.corr_factor = 1.0
        self.column = 0

    def next(self):
        try:
            self.guessed_delta = self.prev[-self.stride] - self.prev[-self.stride * 2]
            result = self.prev[-self.stride] + (self.guessed_delta * self.corr_factor)
            log("P: prev={} delta={} corr_factor={}: result={}".format(self.prev[-self.stride], self.guessed_delta, self.corr_factor, result))
            return result
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
        self.column = (self.column + 1) % self.stride
        if self.column:
            try:
                delta = value - self.prev[-self.stride]
                self.corr_factor = delta / self.guessed_delta
            except (ZeroDivisionError, IndexError):
                self.corr_factor = 1.0
        else:
            self.corr_factor = 1.0
        self.prev.append(value)
        if len(self.prev) > self.stride * 2:
            del self.prev[0]


class MultiPredictor:
    def __init__(self, *predictors):
        self.predictors = predictors
        self.nexts = None
        self.freqs = [[0] * len(predictors) for p in predictors]
        self.prev_best = 0
        self.rescale_limit = len(predictors) ** 2 * 50
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
        self.freqs[self.prev_best][cur_i] += 1
        if cur_i == self.chosen:
            log("P: chosen={} best={} (success!)".format(self.chosen, cur_i))
        else:
            log("P: chosen={} best={} (oops)".format(self.chosen, cur_i))
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


