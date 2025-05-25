Documentation/Lab journal of what I was doing: date and description.

25.5.2025
List of all columns + grouping:

EC = Erythrocye concentrate
## TIME (8 columns --> 1 column):
T_XL -- Date Excel format
T_ISO_T -- Date + Time ISO 
T_DE -- Date European/German format 
T_US_T -- Date + Time US format (mm/dd/yyyy hh:mm)
T_DE_S -- Date European/German SHORT format (dd.mm.yy)
T_US -- Date US format
T_DE_T -- Date + Time European/German format
T_ISO -- Date ISO format

## Patient data (4 columns --> 3 columns)
PAT_BG -- Patient Blood group (A, B, AB, nbn = nicht bestimmt)
PAT_RH -- Patient Rhesus factor (Rh positiv, Rh negativ)
PAT_WARD -- Patient Ward (numbering system)
PAT_BG_RH -- Patient blood group Rhesus factor

# EC data (6 columns --> ?? columns)
EC_BG -- EC Blood group (A,AB, B)
EC_RH -- EC Rhesus factor (Rh positiv, Rh negativ, ???)
EC_TYPE -- EC Type (EKF = Whole bag, EKFX = Split bag)
EC_BG_RH -- EC Blood group rhesus factor, other format (0 +CcD.ee -, B +CcD.ee -, A -ccddee-, und ähnlich)
EC_ID_O_hash
EC_ID_I_hash

# Other data (3 columns --> ?? columns)
ToD  -- Transfusion  or discarded
ToD_N -- Transfused or Discarded
ToD_O -- Transfused or Discarded (Nummer 2?)