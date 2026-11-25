# SPDX-License-Identifier: MIT
"""Exercise the native water update with Blender-style single-precision geometry."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"extension/weft_replay"))
import contact_index
from wet_film import DT, Film, KNIT_WETTING, Liquid, TRANSPORT_SUBSTEPS


class WetTransportTest(unittest.TestCase):
    @unittest.skipIf(contact_index._index is None,"Windows x64 native kernel required")
    def test_native_water_follows_reference_on_a_curved_float32_surface(self):
        rows,columns=23,17
        pitch=.0025
        yy,xx=np.indices((rows,columns))
        arc=(xx.ravel()-columns//2)*pitch/.055
        pos=np.column_stack((.055*np.sin(arc),.055*np.cos(arc),yy.ravel()*pitch)).astype(np.float32)
        area=np.full(rows*columns,pitch*pitch)
        valid=np.ones(len(area),dtype=bool)
        dt=DT/TRANSPORT_SUBSTEPS
        def seeded():
            film=Film((rows,columns),pitch,(valid,valid),area,area,Liquid())
            centre=pos[(rows//2)*columns+columns//2]
            film.dose("outer",np.linalg.norm(pos.astype(np.float64)-centre,axis=1))
            return film
        actual,reference=seeded(),seeded()
        actual_geometry=actual.outer_geometry(pos)
        reference_geometry=tuple(a.astype(np.float64) if a.dtype.kind=="f" else a for a in actual_geometry)
        self.assertEqual(actual_geometry[1].dtype,np.float32)
        for _ in range(10):
            self.assertTrue(contact_index.transport_film(actual,actual_geometry,TRANSPORT_SUBSTEPS,dt,
                                                         KNIT_WETTING.uptake_rate_s))
            for _ in range(TRANSPORT_SUBSTEPS):
                reference.transport(reference_geometry,dt)
        for name in ("outer","knit","momentum","out","evap","outer_to_knit"):
            field=getattr(actual,name)
            self.assertTrue(np.isfinite(field).all(),name)
            np.testing.assert_allclose(field,getattr(reference,name),rtol=2e-6,atol=2e-10,err_msg=name)
        self.assertGreater(actual.knit.sum(),0.)
        self.assertLess(abs(actual.inventory()["balance_error_ml"]),1e-10)


if __name__=="__main__":
    unittest.main()
