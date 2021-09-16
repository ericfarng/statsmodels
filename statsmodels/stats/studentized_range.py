"""Studentized Range distribution, typically used in Tukey's HSD

Created on Wed Sep 15 2021

Author: Eric Farng

Notes
-----

This is a translation of the Pascal code from this paper
http://jaguar.fcav.unesp.br/RME/fasciculos/v25/v25_n1/A8_Daniel.pdf
I included the Pascal code here for completeness.

This python code is same as Pascal code up to around 12 digits, depending on parameters
This python code is same as R code up to around 5 digits, depending on parameters

statsmodels.stats.libqsturng is based on a paper that implements a much faster version
and that is true for this python code, which is much slower than statsmodels.stats.libqsturng
(in this python code, apnorm() is the bottleneck, but I don't immediately see a way to make it faster, fint() is second)

Other differences with statsmodels.stats.libqsturng:
* Pascal code authors validated the correctness of their code in the paper, and is not an approximation
* libqsturng only supports p-values from 0.1 to 0.999
* libqsturng is same as R code _only_ up to 1 or 2 digits
I believe the difference is because libqsturng uses a table of pre-calculated values,
using the same algorithm as R, but the table has different values than R.
Since I don't know the code or the paper for libqsturng, I hesitate to update that table
which might have unexpected results.

"""


import math

_root = [0.993128599185095, 0.963971927277914,
         0.912234428251326, 0.839116971822219, 0.746331906460151, 0.636053680726515,
         0.510867001950827, 0.37370608871542, 0.227785851141645, 0.0765265211334973,
         -0.0765265211334973, -0.227785851141645, -0.37370608871542,
         -0.510867001950827, -0.636053680726515, -0.746331906460151,
         -0.839116971822219, -0.912234428251326, -0.963971927277914,
         -0.993128599185095]
_weight = [0.0176140071391521, 0.0406014298003869,
           0.0626720483341091, 0.0832767415767048, 0.10193011981724, 0.118194531961518,
           0.131688638449177, 0.142096109318382, 0.149172986472604, 0.152753387130726,
           0.152753387130726, 0.149172986472604, 0.142096109318382, 0.131688638449177,
           0.118194531961518, 0.10193011981724, 0.0832767415767048, 0.0626720483341091,
           0.0406014298003869, 0.0176140071391521]


def _apnorm(z):
    """ normal probabilities - accuracy of 1.e-15.

    Parameters
    ----------
    z : float
        number of standard deviation from mean

    Returns
    -------
    float
        The probability

    Notes
    -----
    normal probabilities – accuracy of 1.e-15.
    Z = number of standard deviation from mean
    P, Q = Left and right probabilities from Z. P + Q = 1.
    PDF = the probability density.
    Based upon algorithm 5666 for the error function, from:
    Hart, J.F. et al, 'Computer Approximations', Wiley 1968
    Delphi version: 04/11/2004

    This is similar to scipy.stats.norm.cdf() but is much faster
    """

    p0 = 220.2068679123761E0
    p1 = 221.2135961699311E0
    p2 = 112.0792914978709E0
    p3 = 33.91286607838300E0
    p4 = 6.373962203531650E0
    p5 = 0.7003830644436881E0
    p6 = 0.3526249659989109E-01
    q0 = 440.4137358247522E0
    q1 = 793.8265125199484E0
    q2 = 637.3336333788311E0
    q3 = 296.5642487796737E0
    q4 = 86.78073220294608E0
    q5 = 16.06417757920695E0
    q6 = 1.755667163182642E0
    q7 = 0.8838834764831844E-1
    cutoff = 7.071E0
    root2_pi = 2.506628274631001E0

    zabs = abs(z)
    if zabs > 37.0E0:  # |z| > 37.
        if z > 0.0E0:
            p = 1.0E0
        else:
            p = 0.0E0
    else:  # |z| <= 37
        expntl = math.exp(-0.5E0 * zabs * zabs)
        pdf = expntl / root2_pi
        if zabs < cutoff:  # |z| < cutoff = 10/sqrt(2).
            p = expntl * ((((((p6 * zabs + p5) * zabs + p4) * zabs + p3) * zabs + p2) * zabs + p1) * zabs + p0) / \
                (((((((q7 * zabs + q6) * zabs + q5) * zabs + q4) * zabs + q3) * zabs + q2) * zabs + q1) * zabs + q0)

        else:  # |z| >= cutoff
            p = pdf / (zabs + 1.0E0 / (zabs + 2.0E0 / (zabs + 3.0E0 / (zabs + 4.0E0 / (zabs + 0.65E0)))))
        if z >= 0.0E0:
            q = p
            p = 1.0E0 - q
    return p


# ----------------------------- NORMAL ---------------------------- 
def _apnorminv(p):
    """ Produces the normal deviate Z corresponding to a given lower tail area of P

    Parameters
    ----------
    p : float
        probability

    Returns
    -------
    float
        the cumulative density

    Notes
    -----
    original name: PPND16
    ALGORITHM AS241 APPL. STATIST. (1988) VOL. 37, NO. 3
    Produces the normal deviate Z corresponding to a given lower
    tail area of P Z is accurate to about 1 part in 10**16.
    The hash sums below are the sums of the mantissas of the
    coefficients. They are included for use in checking
    transcription.
    Delphi version

    This is similar to scipy.stats.norm.ppf() but is much faster
    """

    zero = 0.0E0
    one = 1.0E0
    half = 0.5E0
    split1 = 0.425E0
    split2 = 5.0E0
    const1 = 0.180625E0
    const2 = 1.6E0
    # Coefficients for P close to 0.5
    a0 = 3.3871328727963666080E0
    a1 = 1.3314166789178437745E+2
    a2 = 1.9715909503065514427E+3
    a3 = 1.3731693765509461125E+4
    a4 = 4.5921953931549871457E+4
    a5 = 6.7265770927008700853E+4
    a6 = 3.3430575583588128105E+4
    a7 = 2.5090809287301226727E+3
    b1 = 4.2313330701600911252E+1
    b2 = 6.8718700749205790830E+2
    b3 = 5.3941960214247511077E+3
    b4 = 2.1213794301586595867E+4
    b5 = 3.9307895800092710610E+4
    b6 = 2.8729085735721942674E+4
    b7 = 5.2264952788528545610E+3
    # Coefficients for P not close to 0, 0.5 or 1.
    c0 = 1.42343711074968357734E0
    c1 = 4.63033784615654529590E0
    c2 = 5.76949722146069140550E0
    c3 = 3.64784832476320460504E0
    c4 = 1.27045825245236838258E0
    c5 = 2.41780725177450611770E-1
    c6 = 2.27238449892691845833E-2
    c7 = 7.74545014278341407640E-4
    d1 = 2.05319162663775882187E0
    d2 = 1.67638483018380384940E0
    d3 = 6.89767334985100004550E-1
    d4 = 1.48103976427480074590E-1
    d5 = 1.51986665636164571966E-2
    d6 = 5.47593808499534494600E-4
    d7 = 1.05075007164441684324E-9
    # Coefficients for P near 0 or 1.
    e0 = 6.65790464350110377720E0
    e1 = 5.46378491116411436990E0
    e2 = 1.78482653991729133580E0
    e3 = 2.96560571828504891230E-1
    e4 = 2.65321895265761230930E-2
    e5 = 1.24266094738807843860E-3
    e6 = 2.71155556874348757815E-5
    e7 = 2.01033439929228813265E-7
    f1 = 5.99832206555887937690E-1
    f2 = 1.36929880922735805310E-1
    f3 = 1.48753612908506148525E-2
    f4 = 7.86869131145613259100E-4
    f5 = 1.84631831751005468180E-5
    f6 = 1.42151175831644588870E-7
    f7 = 2.04426310338993978564E-15

    q = p - half
    if abs(q) <= split1:
        r = const1 - q * q
        ppnd = q * (((((((a7 * r + a6) * r + a5) * r + a4) * r + a3) * r + a2) * r + a1) * r + a0) / \
            (((((((b7 * r + b6) * r + b5) * r + b4) * r + b3) * r + b2) * r + b1) * r + one)

    else:
        if q < zero:
            r = p
        else:
            r = one - p
        if r <= zero:
            ppnd = zero
        else:
            r = math.sqrt(-math.log(r))
            if r <= split2:
                r = r - const2
                ppnd = (((((((c7 * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0) / \
                       (((((((d7 * r + d6) * r + d5) * r + d4) * r + d3) * r + d2) * r + d1) * r + one)
            else:
                r = r - split2
                ppnd = (((((((e7 * r + e6) * r + e5) * r + e4) * r + e3) * r + e2) * r + e1) * r + e0) / \
                       (((((((f7 * r + f6) * r + f5) * r + f4) * r + f3) * r + f2) * r + f1) * r + one)
            if q < zero:
                ppnd = -ppnd
    return ppnd


def _lngammaf(z):
    """ Lanczos' approximation for ln(gamma) and z > 0

    Parameters
    ----------
    z : float

    Returns
    -------
    float

    Notes
    -----
    Uses Lanczos' approximation for ln(gamma) and z > 0. Reference:
    Lanczos, C. 'A precision approximation of the gamma
    function', J. SIAM Numer. Anal., B, 1, 86-96, 1964.
    Accuracy: About 14 significant digits except for small regions
    in the vicinity of 1 and 2.
    Programmer: Alan Miller - CSIRO Division of Mathematics & Statistics
    Latest revision - 17 April 1988
    Delphi version: Date: 04/11/2004
    """
    a = [0.9999999999995183E0,
         676.5203681218835E0,
         -1259.139216722289E0,
         771.3234287757674E0,
         -176.6150291498386E0,
         12.50734324009056E0,
         -0.1385710331296526E0,
         0.9934937113930748E-05,
         0.1659470187408462E-06]
    lnsqrt2pi = 0.9189385332046727E0
    if z <= 0.0E0:
        raise Exception("need: Z > 0")

    lngamma = 0.0E0
    tmp = z + 7.0E0
    for j in range(8, 0, -1):
        lngamma = lngamma + a[j] / tmp
        tmp = tmp - 1.0E0
    lngamma = lngamma + a[0]
    lngamma = math.log(lngamma) + lnsqrt2pi - (z + 6.5E0) + (z - 0.5E0) * math.log(z + 6.5E0)
    return lngamma


# ------------------------ Ln_da_Gama ------------------------
def _prange_v_inf(w, r):
    def fint(ww, yii, aii, bii, rr):
        yyi = (bii - aii) * yii + bii + aii
        return math.exp(-yyi * yyi * 0.125) * (_apnorm(yyi * 0.5) - _apnorm((yyi - 2 * ww) * 0.5)) ** (rr - 1)

    def gauss_legre_quadrature(ww, aii, bii, rr, a, b, n):
        jfirst = 0
        jlast = n
        c = (b - a) * 0.5
        d = (b + a) * 0.5
        weight_sum = 0.0
        for j in range(jfirst, jlast):
            if _root[j] == 0.0:
                weight_sum = weight_sum + _weight[j] * fint(ww, d, aii, bii, rr)
            else:
                weight_sum = weight_sum + _weight[j] * (fint(ww, _root[j] * c + d, aii, bii, rr))

        return c * weight_sum

    if w <= 0:
        return 0.0

    if w <= 3:
        k = 3.0
    else:
        k = 2.0
    # inicializando valor de ai p/ i=1
    ai = w / 2.0
    ii = 1
    bi = ((k - ii) * (w / 2.0) + 8 * ii) / float(k)
    soma = 0
    for i in range(1, int(round(k)) + 1):  # loop para soma externa de i = 1 a k
        ii = i
        soma = soma + ((bi - ai) / 2.0) * gauss_legre_quadrature(w, ai, bi, r, -1.0, +1.0, 20)
        ai = bi
        if i + 1 == round(k):
            bi = 8
        else:
            bi = ((k - ii - 1) * (w / 2.0) + 8 * (ii + 1)) / float(k)

    soma = soma * 2 * r / math.sqrt(2 * math.pi)
    soma = soma + (math.exp(1)) ** (r * math.log(2 * _apnorm(w / 2.0) - 1))
    return soma


def prange(q, r, v, ci=1):
    """ Distribution function for the studentized range distribution
    given quantile q for r samples and v degrees of freedom.

    Parameters
    ----------
    q : float
        Quantile for Studentized Range
    r : int
        Number of samples
    v : float
        Degrees of freedom

    Returns
    -------
    p : float
        Cumulative probability
    """
    def f26(qq, za, aii, cc, rr, vv):
        yyi = (za * ll + 2 * aii * ll + ll)
        aux1 = _prange_v_inf(math.sqrt(yyi / 2.0) * qq, rr)
        if aux1 == 0:
            aux1 = 1E-37
        aux = cc * math.log(aux1) + math.log(ll) + (vv / 2.0) * math.log(vv) + \
            (-yyi * vv / 4.0) + (vv / 2.0 - 1) * math.log(yyi) - (vv * math.log(2) + _lngammaf(vv / 2.0))
        if abs(aux) >= 1E30:
            return 0
        else:
            return math.exp(aux)

    def gausslegdquad(qq, aii, rr, cii, aa, b, n):
        jfirst = 0
        jlast = n
        cmm = (b - aa) / 2.0
        d = (b + aa) / 2.0
        weight_sum = 0.0
        for j in range(jfirst, jlast):

            if _root[j] == 0.0:
                weight_sum = weight_sum + _weight[j] * f26(qq, d, aii, cii, rr, v)
            else:
                weight_sum = weight_sum + _weight[j] * (f26(qq, _root[j] * cmm + d, aii, cii, rr, v))
        return cmm * weight_sum

    precis = 1E-10
    if v == 1:
        if r < 10:
            ll = 1 + 1 / float(2 * r + 3)
        elif r <= 100:
            ll = 1.0844 + (1.119 - 1.0844) / 90.0 * (r - 10)
        else:
            ll = 1.119 + 1 / float(r)
    elif v == 2:
        ll = 0.968
    elif v <= 100:
        ll = 1
    elif v <= 800:
        ll = 1 / 2.0
    elif v <= 5000:
        ll = 1 / 4.0
    else:
        ll = 1 / 8.0

    # if v>25000 use (H(q))^c as approximation to the probability
    # I added gausslegdquad(q, 0, r, ci, -1.0, +1.0, 20) == 0
    # if gausslegdquad() returns zero, then auxprob = 0, then divide by zero error
    # it looks like gausslegdquad() is zero when v is large, so I added the condition here.
    # validated this through some test cases in tukey_test.py
    if v > 25000 or gausslegdquad(q, 0, r, ci, -1.0, +1.0, 20) == 0:
        return _prange_v_inf(q, r) ** ci
    else:
        auxprob = 0

    found = False
    a = 0
    probinic = 0
    while not found:
        auxprob = auxprob + gausslegdquad(q, a, r, ci, -1.0, +1.0, 20)
        if abs(auxprob - probinic) / auxprob <= precis:
            found = True
        else:
            probinic = auxprob
        a = a + 1

    return auxprob


def _qtrngo(p, v, r):
    """ Calculate a initial percentile P from studentized range dist

    Parameters
    ----------
    p : float
        probability
    v : float
        degrees of freedom
    r : int
        number of samples

    Returns
    -------
    float

    Notes
    -----
    algorithm AS 190.2 Appl. Stat. (1983) vol. 32 no.2
    Calculate a initial percentile P from studentized range dist. with v DF
    and r samples, and cumulative probabilities: P [0.80..0.995]
    uses normal inverse functions
    """
    vmax = 120.0
    half = 0.5
    one = 1.0
    four = 4.0
    c1 = 0.8843
    c2 = 0.2368
    c3 = 1.214
    c4 = 1.208
    c5 = 1.4142

    # inic_val
    t = _apnorminv(half + half * p)
    if v < vmax:
        t = t + (t * t * t + t) / float(v) / four
    q = c1 - c2 * t
    if v < vmax:
        q = q - c3 / float(v) + c4 * t / float(v)
    return t * (q * math.log(r - one) + c5)


def qrange(p, r, v, ci=1):
    """ Quantile function of the studentized range distribution
    for probability p with r samples and v degrees of freedom

    Parameters
    ----------
    p : float
        Cumulative probability
    r : int
        Number of samples
    v : float
        degrees of freedom

    Returns
    -------
    q : float
        Quantile for given probability p

    Notes
    -----
    Adapted from Algorithm AS 190.1 Appl. Stat. (1983) vol. 32 no.2
    approximate the percentile P from studentized range dist. with v DF
    and r samples, and cumulative probabilities: P [0.0..1.00]
    uses functions: normal inverse, normal pdf, prange and qtrngo
    """
    jmax = 28
    pcut = 1E-8
    one = 1.0
    two = 2.0
    five = 5.0

    # verifying initial values
    if (v < one) or (r < two):
        raise Exception("need: v >= 1 and r >= 2")

    if p <= 0 or p >= 1.0:
        raise Exception("need: 0 < p < 1.0")

    # obtaining initial values 
    q1 = _qtrngo(p, v, r)

    while True:
        p1 = prange(q1, r, v, ci)
        if p1 > p:
            q1 = q1 - 0.4
        if q1 < 0:
            q1 = 0.1
        if p1 < p:
            break

    aux = q1
    if abs(p1 - p) < pcut:
        raise Exception("bad: abs(P1 - p) < pcut")

    q2 = q1 + 0.5
    while True:
        p2 = prange(q2, r, v, ci)

        if p2 < p:
            q2 = q2 + 0.4
        if q2 < 0:
            q2 = 1
        if p2 > p:
            break
    if q2 < q1:
        q2 = q1 + 0.01

    # Refiningtheprocedure 
    j = 2
    while j <= jmax:
        p2 = prange(q2, r, v, ci)
        e1 = p1 - p
        e2 = p2 - p
        if e2 - e1 != 0:
            aux = (e2 * q1 - e1 * q2) / float(e2 - e1)
        if abs(e1) < abs(e2):
            if abs(p1 - p) < pcut * five:
                j = jmax + 2
            q1 = aux
            p1 = prange(q1, r, v, ci)
        else:
            q1 = q2
            p1 = p2
            q2 = aux
        j = j + 1
    return aux

"""
(*
This code is directly from here.
http://jaguar.fcav.unesp.br/RME/fasciculos/v25/v25_n1/A8_Daniel.pdf
At the bottom, is the test case I used to compare implementations


unit ptukey;

interface

Function PRange_v_inf(w, r: Extended; var ifault: Longint): Extended;
Function PRange(q, r, v, ci: Extended; var ifault: Longint): Extended;
Function qrange(p, r, v, ci: Extended; Var ifault: Longint): Extended;
Function apnorm(Z: Extended): Extended;
Function apnorminv(p: Extended): Extended;
function lngammaf(Z: Extended; var ier: Longint): Extended;

implementation
*)


uses math;

Const
  Root: Array [1 .. 20] of Extended = (0.993128599185095, 0.963971927277914,
    0.912234428251326, 0.839116971822219, 0.746331906460151, 0.636053680726515,
    0.510867001950827, 0.37370608871542, 0.227785851141645, 0.0765265211334973,
    -0.0765265211334973, -0.227785851141645, -0.37370608871542,
    -0.510867001950827, -0.636053680726515, -0.746331906460151,
    -0.839116971822219, -0.912234428251326, -0.963971927277914,
    -0.993128599185095);
  Weight: Array [1 .. 20] of Extended = (0.0176140071391521, 0.0406014298003869,
    0.0626720483341091, 0.0832767415767048, 0.10193011981724, 0.118194531961518,
    0.131688638449177, 0.142096109318382, 0.149172986472604, 0.152753387130726,
    0.152753387130726, 0.149172986472604, 0.142096109318382, 0.131688638449177,
    0.118194531961518, 0.10193011981724, 0.0832767415767048, 0.0626720483341091,
    0.0406014298003869, 0.0176140071391521);

Function apnorm(Z: Extended): Extended;
_Const
  P0 = 220.2068679123761E0;
  P1 = 221.2135961699311E0;
  P2 = 112.0792914978709E0;
  P3 = 33.91286607838300E0;
  P4 = 6.373962203531650E0;
  P5 = 0.7003830644436881E0;
  P6 = 0.3526249659989109E-01;
  Q0 = 440.4137358247522E0;
  Q1 = 793.8265125199484E0;
  Q2 = 637.3336333788311E0;
  Q3 = 296.5642487796737E0;
  Q4 = 86.78073220294608E0;
  Q5 = 16.06417757920695E0;
  Q6 = 1.755667163182642E0;
  Q7 = 0.8838834764831844E-1;
  CUTOFF = 7.071E0;
  ROOT2PI = 2.506628274631001E0;
var
  zabs, expntl: Extended;
  p, q, pdf: Extended;
begin
  zabs := abs(Z);
  if (zabs > 37.0E0) then // |z| > 37.
  begin
    pdf := 0.0E0;
    if (Z > 0.0E0) then
    begin
      p := 1.0E0;
      q := 0.0E0
    end
    else
    begin
      p := 0.0E0;
      q := 1.0E0
    end
  end
  else
  begin // |z| <= 37
    expntl := exp(-0.5E0 * zabs * zabs);
    pdf := expntl / ROOT2PI;
    // |z| < cutoff = 10/sqrt(2).
    if (zabs < CUTOFF) then
      p := expntl * ((((((P6 * zabs + P5) * zabs + P4) * zabs + P3) * zabs + P2)
        * zabs + P1) * zabs + P0) /
        (((((((Q7 * zabs + Q6) * zabs + Q5) * zabs + Q4) * zabs + Q3) * zabs +
        Q2) * zabs + Q1) * zabs + Q0)
      // |z| >= cutoff
    else
      p := pdf / (zabs + 1.0E0 / (zabs + 2.0E0 / (zabs + 3.0E0 / (zabs + 4.0E0 /
        (zabs + 0.65E0)))));
    if (Z < 0.0E0) then
      q := 1.0E0 - p
    else
    begin
      q := p;
      p := 1.0E0 - q
    end;
  end; // z<=37
  apnorm := p;
end;

(* ----------------------------- NORMAL ---------------------------- *)
Function apnorminv(p: Extended): Extended;
// original name: PPND16
// ALGORITHM AS241 APPL. STATIST. (1988) VOL. 37, NO. 3
// Produces the normal deviate Z corresponding to a given lower
// tail area of P; Z is accurate to about 1 part in 10**16.
// The hash sums below are the sums of the mantissas of the
// coefficients. They are included for use in checking
// transcription.
// Delphi version
Const
  ZERO = 0.0E0;
  ONE = 1.0E0;
  HALF = 0.5E0;
  SPLIT1 = 0.425E0;
  SPLIT2 = 5.0E0;
  CONST1 = 0.180625E0;
  CONST2 = 1.6E0;
  // Coefficients for P close to 0.5
  A0 = 3.3871328727963666080E0;
  A1 = 1.3314166789178437745E+2;
  A2 = 1.9715909503065514427E+3;
  A3 = 1.3731693765509461125E+4;
  A4 = 4.5921953931549871457E+4;
  A5 = 6.7265770927008700853E+4;
  A6 = 3.3430575583588128105E+4;
  A7 = 2.5090809287301226727E+3;
  B1 = 4.2313330701600911252E+1;
  B2 = 6.8718700749205790830E+2;
  B3 = 5.3941960214247511077E+3;
  B4 = 2.1213794301586595867E+4;
  B5 = 3.9307895800092710610E+4;
  B6 = 2.8729085735721942674E+4;
  B7 = 5.2264952788528545610E+3;
  // Coefficients for P not close to 0, 0.5 or 1.
  C0 = 1.42343711074968357734E0;
  C1 = 4.63033784615654529590E0;
  C2 = 5.76949722146069140550E0;
  C3 = 3.64784832476320460504E0;
  C4 = 1.27045825245236838258E0;
  C5 = 2.41780725177450611770E-1;
  C6 = 2.27238449892691845833E-2;
  C7 = 7.74545014278341407640E-4;
  D1 = 2.05319162663775882187E0;
  D2 = 1.67638483018380384940E0;
  D3 = 6.89767334985100004550E-1;
  D4 = 1.48103976427480074590E-1;
  D5 = 1.51986665636164571966E-2;
  D6 = 5.47593808499534494600E-4;
  D7 = 1.05075007164441684324E-9;
  // Coefficients for P near 0 or 1.
  E0 = 6.65790464350110377720E0;
  E1 = 5.46378491116411436990E0;
  E2 = 1.78482653991729133580E0;
  E3 = 2.96560571828504891230E-1;
  E4 = 2.65321895265761230930E-2;
  E5 = 1.24266094738807843860E-3;
  E6 = 2.71155556874348757815E-5;
  E7 = 2.01033439929228813265E-7;
  F1 = 5.99832206555887937690E-1;
  F2 = 1.36929880922735805310E-1;
  F3 = 1.48753612908506148525E-2;
  F4 = 7.86869131145613259100E-4;
  F5 = 1.84631831751005468180E-5;
  F6 = 1.42151175831644588870E-7;
  F7 = 2.04426310338993978564E-15;
var
  ppnd, q, r: Extended;
  ifault: Longint;
begin
  ifault := 0;
  q := p - HALF;
  if (abs(q) <= SPLIT1) then
  begin
    r := CONST1 - q * q;
    ppnd := q * (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r
      + A1) * r + A0) /
      (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1)
      * r + ONE)
  end
  else
  begin
    if (q < ZERO) then
      r := p
    else
      r := ONE - p;
    if (r <= ZERO) then
    begin
      ifault := 1;
      ppnd := ZERO
    end
    else
    begin
      r := sqrt(-ln(r));
      if (r <= SPLIT2) then
      begin
        r := r - CONST2;
        ppnd := (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r
          + C1) * r + C0) /
          (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1)
          * r + ONE)
      end
      else
      begin
        r := r - SPLIT2;
        ppnd := (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r
          + E1) * r + E0) /
          (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1)
          * r + ONE)
      end;
      if (q < ZERO) then
        ppnd := -ppnd;
    end;
  end;
  apnorminv := ppnd
end; // norminv

function lngammaf(Z: Extended; var ier: Longint): Extended;
// Uses Lanczos’ approximation for ln(gamma) and z > 0. Reference:
// Lanczos, C. 'A precision approximation of the gamma
// function', J. SIAM Numer. Anal., B, 1, 86-96, 1964.
// Accuracy: About 14 significant digits except for small regions
// in the vicinity of 1 and 2.
// Programmer: Alan Miller - CSIRO Division of Mathematics & Statistics
// Latest revision - 17 April 1988
// Delphi version: Date: 04/11/2004
var
  a: array [1 .. 9] of Extended;
  lnsqrt2pi, tmp, lngamma: Extended;
  j: Longint;
Begin
  a[1] := 0.9999999999995183E0;
  a[2] := 676.5203681218835E0;
  a[3] := -1259.139216722289E0;
  a[4] := 771.3234287757674E0;
  a[5] := -176.6150291498386E0;
  a[6] := 12.50734324009056E0;
  a[7] := -0.1385710331296526E0;
  a[8] := 0.9934937113930748E-05;
  a[9] := 0.1659470187408462E-06;
  lnsqrt2pi := 0.9189385332046727E0;
  if (Z <= 0.0E0) then
  begin
    ier := 1;
    exit;
  end;
  ier := 0;
  lngamma := 0.0E0;
  tmp := Z + 7.0E0;
  for j := 9 downto 2 do
  begin
    lngamma := lngamma + a[j] / tmp;
    tmp := tmp - 1.0E0
  end;
  lngamma := lngamma + a[1];
  lngamma := ln(lngamma) + lnsqrt2pi - (Z + 6.5E0) + (Z - 0.5E0) *
    ln(Z + 6.5E0);
  lngammaf := lngamma;
end;

(* ------------------------ Ln_da_Gama ------------------------ *)
Function PRange_v_inf(w, r: Extended; var ifault: Longint): Extended;
var
  k, ai, bi, soma, ii: Extended;
  i: Longint;
  function fint(w, yii, aii, bii, r: Extended): Extended;
  var
    yyi: Extended;
  begin
    yyi := (bii - aii) * yii + bii + aii;
    fint := Power(exp(1), -yyi * yyi / 8) *
      Power((apnorm(yyi / 2) - apnorm((yyi - 2 * w) / 2)), r - 1);
  end;
  function GaussLegendreQuadrature(w, yii, aii, bii, r: Extended;
    const a, b: double; const n: Longint; var ifault: Longint): Extended;
  var
    c, d, sum: Extended;
    j, jfirst, jlast: Longint;
  begin
    jfirst := 1;
    jlast := n;
    c := (b - a) / 2.0;
    d := (b + a) / 2.0;
    sum := 0.0;
    for j := jfirst to jlast do
    begin
      if Root[j] = 0.0 then
        sum := sum + Weight[j] * fint(w, d, aii, bii, r)
      else
        sum := sum + Weight[j] * (fint(w, Root[j] * c + d, aii, bii, r));
    end;
    GaussLegendreQuadrature := c * sum
  end { gausslegendrequadrature };

begin
  if w <= 0 then
  begin
    PRange_v_inf := 0;
    exit
  end;
  if w <= 3 then
    k := 3.0
  else
    k := 2.0;
  // inicializando valor de ai p/ i=1
  ai := w / 2;
  ii := 1;
  bi := ((k - ii) * (w / 2) + 8 * ii) / k;
  soma := 0;
  for i := 1 to round(k) do // loop para soma externa de i = 1 a k
  begin
    ii := i;
    soma := soma + ((bi - ai) / 2) * GaussLegendreQuadrature(w, 0.0, ai, bi, r,
      -1.0, +1.0, 20, ifault);
    ai := bi;
    if i + 1 = round(k) then
      bi := 8
    else
      bi := ((k - ii - 1) * (w / 2) + 8 * (ii + 1)) / k;
  end;
  soma := soma * 2 * r / sqrt(2 * Pi);
  soma := soma + Power(exp(1), r * ln(2 * apnorm(w / 2) - 1));
  PRange_v_inf := soma
end;

function PRange(q, r, v, ci: Extended; var ifault: Longint): Extended;
var
  precis, a, auxprob, L, probinic: Extended;
  found: Boolean;

  function f26(q, za, aii, c, r, v: Extended): Extended;
  Var
    yyi, aux, aux1: Extended;
  begin
    yyi := (za * L + 2 * aii * L + L);
    aux1 := PRange_v_inf(sqrt(yyi / 2) * q, r, ifault);
    if aux1 = 0 then
      aux1 := 1E-37;
    aux := c * ln(aux1) + ln(L) + (v / 2) * ln(v) + (-yyi * v / 4) + (v / 2 - 1)
      * ln(yyi) - (v * ln(2) + lngammaf(v / 2, ifault));
    if abs(aux) >= 1E30 then
      f26 := 0
    else
      f26 := exp(aux);
  end;
  function gausslegdquad(q, yii, aii, r, ci: Extended; const a, b: double;
    const n: Longint; var ifault: Longint): Extended;
  var
    cmm, d, sum: Extended;
    j, jfirst, jlast: Longint;
  begin
    jfirst := 1;
    jlast := n;
    cmm := (b - a) / 2.0;
    d := (b + a) / 2.0;
    sum := 0.0;
    for j := jfirst to jlast do
    begin
      if Root[j] = 0.0 then
        sum := sum + Weight[j] * f26(q, d, aii, ci, r, v)
      else
        sum := sum + Weight[j] * (f26(q, Root[j] * cmm + d, aii, ci, r, v));
    end;
    gausslegdquad := cmm * sum
  end { gausslegendrequadrature };

begin
  precis := 1E-10;
  ifault := 0;
  if v = 1 then
  begin
    if r < 10 then
      L := 1 + 1 / (2 * r + 3)
    else if r <= 100 then
      L := 1.0844 + (1.119 - 1.0844) / 90 * (r - 10)
    else
      L := 1.119 + 1 / r;
  end
  else if (v = 2) then
    L := 0.968
  else if v <= 100 then
    L := 1
  else if v <= 800 then
    L := 1 / 2
  else if v <= 5000 then
    L := 1 / 4
  else
    L := 1 / 8; // if v>25000 use (H(q))^c as approximation to the probability
  if v > 25000 then
  begin
    PRange := Power(PRange_v_inf(q, r, ifault), ci);
    exit
  end
  else
    auxprob := 0;
  found := false;
  a := 0;
  probinic := 0;
  while not found do
  begin
    auxprob := auxprob + gausslegdquad(q, 0, a, r, ci, -1.0, +1.0, 20, ifault);
    if abs(auxprob - probinic) / auxprob <= precis then
      found := true
    else
      probinic := auxprob;
    a := a + 1;
  end;
  PRange := auxprob;
end;

(* **************************************************************** *)
(* algorithm AS 190.2 Appl. Stat. (1983) vol. 32 no.2 *)
(* Calculate a initial percentile P from studentized range dist. with v DF *)
(* and r samples, and cumulative probabilities: P [0.80..0.995] *)
(* uses normal inverse functions *)
(* **************************************************************** *)
Function qtrngo(p, v, r: Extended; Var ifault: Longint): Extended;
Var
  q, t, Vmax, HALF, ONE, four, C1, C2, C3, C4, C5: Extended;
  Procedure inic_val;
  Begin
    Vmax := 120.0;
    HALF := 0.5;
    ONE := 1.0;
    four := 4.0;
    C1 := 0.8843;
    C2 := 0.2368;
    C3 := 1.214;
    C4 := 1.208;
    C5 := 1.4142
  End;

Begin
  inic_val;
  t := apnorminv(HALF + HALF * p);
  if (v < Vmax) then
    t := t + (t * t * t + t) / v / four;
  q := C1 - C2 * t;
  if (v < Vmax) then
    q := q - C3 / v + C4 * t / v;
  qtrngo := t * (q * ln(r - ONE) + C5)
end;
(* **********************************************************************
  ** *)
(* Adapted from Algorithm AS 190.1 Appl. Stat. (1983) vol. 32 no.2 *)
(* approximate the percentile P from studentized range dist. with v DF *)
(* and r samples, and cumulative probabilities: P [0.0..1.00] *)

(* uses functions: normal inverse, normal pdf, prange and qtrngo *)
function qrange(p, r, v, ci: Extended; var ifault: Longint): Extended;
var
  jmax, pcut, ONE, two, five: Extended;
  j, Q1, Q2, P1, P2: Extended;
  aux, E1, E2: Extended;
  procedure valores_inic;
  begin
    jmax := 28;
    pcut := 1E-8;
    ONE := 1.0;
    two := 2.0;
    five := 5.0;
  end;

begin
  valores_inic;
  (* verifying initial values *)
  ifault := 0;
  if (v < ONE) or (r < two) then
    ifault := 1;
  if ifault <> 0 then
  begin
    qrange := 0;
    exit;
  end
  else
  begin
    (* obtaining initial values *)
    Q1 := qtrngo(p, v, r, ifault);
    if ifault <> 0 then
    begin
      if ifault <> 0 then
        ifault := 9;
      qrange := 0;
      exit;
    end;
    repeat
      P1 := PRange(Q1, r, v, ci, ifault);
      if P1 > p then
        Q1 := Q1 - 0.4;
      if Q1 < 0 then
        Q1 := 0.1;
    until P1 < p;
    if ifault <> 0 then
    begin
      if ifault <> 0 then
        ifault := 9;
      qrange := 0;
      exit;
    end;
    aux := Q1;
    if abs(P1 - p) < pcut then
    begin
      if ifault <> 0 then
        ifault := 9;
      qrange := Q1;
      exit;
    end;
    Q2 := Q1 + 0.5;
    repeat
      P2 := PRange(Q2, r, v, ci, ifault);

      if P2 < p then
        Q2 := Q2 + 0.4;
      if Q2 < 0 then
        Q2 := 1;
    until P2 > p;
    if Q2 < Q1 then
      Q2 := Q1 + 0.01;
    if ifault <> 0 then
    begin
      if ifault <> 0 then
        ifault := 9;
      qrange := 0;
      exit;
    end;
    (* Refiningtheprocedure *)
    j := 2;
    while j <= jmax do
    begin
      P2 := PRange(Q2, r, v, ci, ifault);
      if ifault <> 0 then
      begin
        if ifault <> 0 then
          ifault := 9;
        j := jmax + 1;
      end
      else
      begin
        E1 := P1 - p;
        E2 := P2 - p;
        if E2 - E1 <> 0 then
          aux := (E2 * Q1 - E1 * Q2) / (E2 - E1);
        if abs(E1) < abs(E2) then
        begin
          if (abs(P1 - p) < pcut * five) then
          begin
            if ifault <> 0 then
              ifault := 9;
            j := jmax + 2
          end;
          Q1 := aux;
          P1 := PRange(Q1, r, v, ci, ifault);
        end
        else
        begin
          Q1 := Q2;
          P1 := P2;
          Q2 := aux;
        end;
      end;
      j := j + 1;
    end;
    qrange := aux
  end
end;

var ifault:LongInt;
p, q, v: Real;
r: Integer;
p_list: Array[1..9] of Real = (0.02, 0.21, 0.53, 0.82, 0.89, 0.93, 0.97, 0.992, 0.9997);
q_list: Array[1..6] of Real = (1.3, 2.7, 4.7, 10.5, 18.4, 31.1);
r_list: Array[1..7] of Integer = (2, 3, 4, 5, 8, 13, 23);
v_list: Array[1..6] of Real = (3.6, 5.4, 7.9, 13.6, 17.3, 31.6);

begin
  writeln ('Hello World');
  for p in p_list do
  begin
      for r in r_list do
      begin
          for v in v_list do
          begin
              writeln(p:4:4,',',r:2,',',v:4:1,':',qrange(p , r,  v, 1, ifault):18:12)
          end;
      end;
  end;

  for q in q_list do
  begin
      for r in r_list do
      begin
          for v in v_list do
          begin
              writeln(q:4:1,',',r:2,',',v:4:1,':',PRange(q , r,  v, 1, ifault):18:12)
          end;
      end;
  end;

  writeln ('Done!');
end.
"""