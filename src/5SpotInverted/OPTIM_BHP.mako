
-------------------------------------------------------------------------------
-- DATAFILE FOR ECLIPSE TESTING
-------------------------------------------------------------------------------


---------------------------- Runspec Section ----------------------------------
--NOECHO

RUNSPEC

TITLE
 50x50x1=2,500 Eclipse test example

DIMENS
   50   50    1  /

-- Phases present

OIL

WATER

GAS

DISGAS

-- Units

METRIC

-- Table dimension

TABDIMS

-- NoSatTabl     MaxNodesSatTab  MaxFIPReg      MaxSatEndpointsDepthTab
--       NoPVTTab       MaxPressNodes    MaxRsRvNodes
     1       1      20      200       1      200       1    /


-- Well dimension

WELLDIMS
-- MaxNo  MaxPerf  MaxGroup MaxWell/Group
    5     1        5       5  /


START
  1 'JAN' 1970 /


--FMTOUT

--UNIFIN
--UNIFOUT

--NOSIM

NSTACK
 10 /

------------------------------- Grid Section ----------------------------------

GRID

-- Including the indiviual grid file

INCLUDE
 '../50X50X1.COORD' /

INCLUDE
 '../50X50X1.ZCORN' /

--INCLUDE
-- 'TRUE_PORO' /

PORO
2500*0.2
/

-- corr = np.array([40]); std = np.array([1.5]); mean = np.log(500) - std**2/2; permx = mean + fast_gaussian(dim, std, corr)
INCLUDE
 '../PERMX' /

--PERMX
--2500*500
--/

COPY
 PERMX PERMY /
 PERMX PERMZ /
/

MULTIPLY
 PERMZ 0.001 /
/

--GRIDFILE
-- 2 1 /

--NOGGF

INIT

NEWTRAN


------------------------------- Edit Section ----------------------------------


------------------------------ Properties Section -----------------------------


PROPS

ROCK
-- RefPressure          Compressibility
-- for PoreVol Calc
--BARSA                 1/BARSA
  200                   1.450E-05 /

INCLUDE
 '../ALL.PVO' /  

INCLUDE
 '../ALL.RCP' /



------------------------------- Regions Section -------------------------------


------------------------------ Solution Section -------------------------------

SOLUTION

EQUIL
2000.000 200.00 2280.00  .000  2000.000  .000     1      0       0 /

PBVD
 1    10 
 1000 10 /

--RPTSOL
-- RESTART /

--RPTRST
-- BASIC=2 /


------------------------------- Summary Section -------------------------------

SUMMARY

------------------------------------------------
--Output of production data/pressure  for FIELD:
------------------------------------------------

FOPR
FWPR
FLPR
FLPT
FOPT
FGPT
FWPT
FPR

-------------------------------------------------
-- Gas and oil in place:
-------------------------------------------------

FOIP
FGIP

-----------------------------------------
--Output of production data for all wells:
-----------------------------------------
WOPR
 /
WWPR
 /
WGPR
 /
WWCT
 /
WGOR
 /
WBHP
  /
WTHP
 /
WWIR
/

FVIR
FVPR
RPTONLY

RUNSUM

SEPARATE

RPTSMRY
 1  /

DATE


TCPU

------------------------------ Schedule Section -------------------------------

SCHEDULE

SKIPREST

RPTRST
 BASIC=2 DEN/

WELSPECS
 INJ1 G1 2 2 2000 WATER /
 INJ2 G1 49 2 2000 WATER /
 INJ3 G1 2 49 2000 WATER /
 INJ4 G1 49 49 2000 WATER /
 PROD G1 25 25 2000 WATER /
/

COMPDAT
--Name I J K1 K2 STATUS 2* RW
 INJ1 2 2 1 1   OPEN   2* 0.25 /
 INJ2 49 2 1 1 OPEN 2* 0.25 /
 INJ3 2 49 1 1 OPEN 2* 0.25 /
 INJ4 49 49 1 1 OPEN 2* 0.25 /
 PROD 25 25 1 1 OPEN 2* 0.25 /
/

WCONPROD
PROD   OPEN BHP 5* 150 /
/

%for p in range(100):  
${"WCONINJE"}
${"INJ1     WAT 1* BHP 2* {} /".format(injbhp[4*p])}
${"INJ2     WAT 1* BHP 2* {} /".format(injbhp[4*p+1])}
${"INJ3     WAT 1* BHP 2* {} /".format(injbhp[4*p+2])}
${"INJ4     WAT 1* BHP 2* {} /".format(injbhp[4*p+3])}
${"/"}

${"TSTEP"}
${" 30 /"} 

%endfor

END


