-- *------------------------------------------*
-- *                                          *
-- * TinyBox: three injectors, three producers
-- *                                          *
-- *------------------------------------------*
RUNSPEC

TITLE
 TINY BOX MODEL

OIL
WATER
GAS
DISGAS

METRIC

TABDIMS
-- NTSFUN  NTPVT  NSSFUN  NPPVT  NTFIP  NRPVT  NTENDP
     1       1      35      30     5     30      1 /

EQLDIMS
-- NTEQUL  NDRXVD  NDPRVD
   1       5       100 /

WELLDIMS
-- NWMAXZ NCWMAX NGMAXZ MWGMAX
    15     15     2      20 /

START
 01 JAN 2022 /

NSTACK
 25 /

NOECHO

GRID
INIT

INCLUDE
 '../model/Grid.grdecl' /

INCLUDE
 '../model/PERMX.INC' /

COPY
 'PERMX'  'PERMY'  /
 'PERMX'  'PERMZ' /
/


PROPS    ===============================================================

INCLUDE
 '../model/pvt.txt' /


SOLUTION ===============================================================

--    DATUM  DATUM   OWC    OWC    GOC    GOC
--    DEPTH  PRESS  DEPTH   PCOW  DEPTH   PCOG
EQUIL
     2355.00 200.46 3000 0.00  2355.0 0.000     /


SUMMARY ================================================================

RUNSUM

FOPT
FGPT
FWPT
FWIT

WWIR
 'INJ1'
 'INJ2'
 'INJ3'
/

WOPR
 'PRO1'
 'PRO2'
 'PRO3'
/

WGPR
 'PRO1'
 'PRO2'
 'PRO3'
/

WWPR
 'PRO1'
 'PRO2'
 'PRO3'
/


SCHEDULE =============================================================

INCLUDE
 '../model/Schdl.sch' /

-- Bottom-hole pressures [bar], constant over the run: the controls
WCONINJE
'INJ1' WATER 'OPEN' BHP 2* ${injbhp[0]} /
'INJ2' WATER 'OPEN' BHP 2* ${injbhp[1]} /
'INJ3' WATER 'OPEN' BHP 2* ${injbhp[2]} /
/

WCONPROD
 'PRO1' 'OPEN' BHP 5* ${prodbhp[0]} /
 'PRO2' 'OPEN' BHP 5* ${prodbhp[1]} /
 'PRO3' 'OPEN' BHP 5* ${prodbhp[2]} /
/

-- Ten steps of 400 days
TSTEP
10*400 /

END
