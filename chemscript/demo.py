# ChemScript Demo
from ChemScript20 import *


m = StructureData()
m = StructureData.LoadFile("demo.cdx")

print(
    """\nWhen loading data or files, you can specify a mimetype.  This will speed the loading because ChemScript will not need to determine the data type from the file contents."""
)

print(""">>>m = StructureData.LoadData('C=CCO', 'smiles')""")
m = StructureData.LoadData("C=CCO", "smiles")
m.List()
print(m)

for types in StructureData.MimeTypes():
    print(types)


print(""">>>print(m.WriteData('smiles'))""")
print(m.WriteData("smiles"))


print("""		print(a.Name, a.GetCartesian().x, a.GetCartesian().y, a.GetCartesian().z)""")
for a in m.Atoms:
    print(a.Name, a.GetCartesian().x, a.GetCartesian().y, a.GetCartesian().z)


print("""		print(b.Atom1.Name, b.Atom2.Name, b.Order.Name)""")
for b in m.Bonds:
    print(b.Atom1.Name, b.Atom2.Name, b.Order.Name)

print("""\nAdd an atom""")

print("""a = m.CreateAtom('O')""")
a = m.CreateAtom("O")

print("""\nAdd a bond""")

print("""b = m.CreateBond(a, m.Atoms[0], None)""")
b = m.CreateBond(a, m.Atoms[0], None)

print("""\nRemove an atom""")

print("""m.RemoveAtom(m.Atoms[0])""")
m.RemoveAtom(m.Atoms[0])

print("""\nRemove a bond""")

print("""m.RemoveBond(m.Bonds[0])""")
m.RemoveBond(m.Bonds[0])

print("""\nRemove a bond and its bonded atoms""")

print("""m.RemoveBond(m.bonds[0], True)""")
m.RemoveBond(m.Bonds[0], True)


print(m.ChemicalName())


print(""">>>m.ConvertTo3DStructure()""")
m.ConvertTo3DStructure()
print(""">>>m.List(True, True, True)""")
m.List(True, True, True)

m.ClearZCoordinates()


print(""">>>m.CleanupStructure()""")
m.CleanupStructure()


print(
    """\n\n\n   6. Minimizing MM2 and energy calculations for 3D structures\n--------------------------------------"""
)


print("""\nMM2 minimization""")

print(""">>>m.Mm2OptimizeGeometry()""")
m.Mm2OptimizeGeometry()

print("""\nCompute MM2 energy""")

print(""">>>print(m.Mm2Energy())""")
print(m.Mm2Energy())


print("""\n\n\n   7. Salt stripping\n--------------------------------------""")


print(""">>>m = StructureData.LoadData('CCCCCN.[Na+].[Na+].c1ccccc1.[O]')""")
m = StructureData.LoadData("CCCCCN.[Na+].[Na+].c1ccccc1.[O]")


st = SaltTable()


print(""">>> st.RegisterWithSmiles("[Na+]", False)""")
st.RegisterWithSmiles("[Na+]", False)

print(""">>> st.RegisterWithSmiles("c1ccccc1", True)""")
st.RegisterWithSmiles("c1ccccc1", True)

print("""st.RegisterWithSmiles("[O]", True)""")
st.RegisterWithSmiles("[O]", True)

print(""">>>list = st.SplitSaltsAndSolvents(m)""")
list = st.SplitSaltsAndSolvents(m)

print(""">>>mainStructureData = list[0]""")
mainStructureData = list[0]

print(""">>>saltPart = list[1]""")
saltPart = list[1]

print(
    """>>>print("main StructureData:", mainStructureData[0].Smiles, " salt:", saltPart[0].Smiles)"""
)
print("main StructureData:", mainStructureData[0].Smiles, " salt:", saltPart[0].Smiles)


print(""">>>target = StructureData.LoadData('C1CCCCC1C')""")
target = StructureData.LoadData("C1CCCCC1C")
print(""">>>query = StructureData.LoadData('C1CCCCC1')""")
query = StructureData.LoadData("C1CCCCC1")

print(
    """\nNow start the atom-by-atom searching.  This results in a map from the target to query"""
)

print(""">>> maps = query.AtomByAtomSearch(target)""")
maps = query.AtomByAtomSearch(target)
print(""">>>for dict in maps:""")
print("""		print("one dict:")""")
print("""		for aa in dict.keys():""")
print("""			r = aa.Name + ' -> ' + dict[aa].Name""")
print("""			print(r)""")
for dct in maps:
    print("one dict:")
    for aa in dct.keys():
        r = aa.Name + " -> " + dct[aa].Name
        print(r)

print(
    """\nIf you don't care about these atom maps and just want the substructure search result, you can use "ContainsSubstructure"."""
)

print(""">>>print(target.ContainsSubstructure(query))""")
print(target.ContainsSubstructure(query))


print(
    """\n\n\n   9. Finding the largest common substructure\n--------------------------------------"""
)


print("""\nLet's load two molecules""")

print(""">>>structure1 = StructureData.LoadData("C1(C)CCCC1CCO")""")
structure1 = StructureData.LoadData("C1(C)CCCC1CCO")
print(""">>>structure2 = StructureData.LoadData("C1CCCC1C")""")
structure2 = StructureData.LoadData("C1CCCC1C")

print("""Use class LargestCommonSubstructure to compute the most common structure\n""")

print(""">>>common = LargestCommonSubstructure.Compute(structure1, structure2)""")
common = LargestCommonSubstructure.Compute(structure1, structure2)
print(""">>>atommap1 = common.AtomMapM1()""")
print(""">>>bondmap1 = common.BondMapM1()""")
print(""">>>atommap2 = common.AtomMapM2()""")
print(""">>>bondmap2 = common.BondMapM2()""")
atommap1 = common.AtomMapM1()
bondmap1 = common.BondMapM1()
atommap2 = common.AtomMapM2()
bondmap2 = common.BondMapM2()
print(""">>>for a in atommap1.keys():""")
print("""		r = a.Name + '->' + atommap1[a].Name + '->' + atommap2[a].Name""")
print("""		print(r)""")
for a in atommap1.keys():
    r = a.Name + "->" + atommap1[a].Name + "->" + atommap2[a].Name
    print(r)


print("""\n\n\n   10. Overlaying structures\n--------------------------------------""")


print("""\n(1). 2D alignment""")


print("""\nFirst let's load two cdx (2D)""")

print(""">>>m = StructureData.LoadFile('m.cdx')""")
m = StructureData.LoadFile("m.cdx")
print(""">>>target = StructureData.LoadFile('target.cdx')""")
target = StructureData.LoadFile("target.cdx")

print("""\nMake 2D alignment""")

print(""">>>print(m.Overlay(target))""")
print(m.Overlay(target))

print("""\nWrite the output into a file""")

print(""">>>m.WriteFile('m_output.cdx')""")
m.WriteFile("m_output.cdx")

print("""\n(2). 3D Overlay""")
print(
    """\nIf the input consists of 3D structures, the overlay will operate on the 3D coordinates"""
)


print(
    """\n\n\n   11. Computing molecular topological properties\n--------------------------------------"""
)


print("""\nFirst create a Topology object""")

print(""">>>top = m.Topology()""")
top = m.Topology()

print("""\nThen you are able to get many topology properties""")

print(""">>>print(top.WienerIndex)""")
print(top.WienerIndex)
print(""">>>print(top.BalabanIndex)""")
print(top.BalabanIndex)
print(""">>>print(top.ShapeCoefficient)""")
print(top.ShapeCoefficient)
print("""\n\n and more ...""")


print(
    """\n\n\n   12. Working with ReactionData objects\n--------------------------------------"""
)


print("""\nCreate a reaction from a SMILES string""")

print(""">>>r = ReactionData.LoadData('C1CCC=CC1>>CCCCCC', 'smiles')""")
r = ReactionData.LoadData("C1CCC=CC1>>CCCCCC", "smiles")

print("""\nLoad a reaction from a file""")

print(""">>>r = ReactionData.LoadFile('reaction.cdx')""")
r = ReactionData.LoadFile("reaction.cdx")
print(""">>>print(r.Formula())""")
print(r.Formula())

print("""\nGet reactants as a StructureData list""")

print(""">>>for rtn in r.Reactants:""")
print(""">>>	print(rtn)""")
for rtn in r.Reactants:
    print(rtn)

print("""\nGet products as a StructureData list""")

print(""">>>for prod in r.Products:""")
print(""">>>	print(prod)""")
for prod in r.Products:
    print(prod)


print(
    """\n\n\n   13. Reading and writing SD files\n--------------------------------------"""
)


print("""\nRead StructureData from an SD file""")

print(""">>>sd = SDFileReader.OpenFile('input.sdf')""")
sd = SDFileReader.OpenFile("input.sdf")
print(""">>>m = sd.ReadNext()""")
print(""">>>while (m != None):""")
print("""		print(m.Formula())""")
print("""		items = m.GetDataItems()""")
print("""		for item in items.keys():""")
print("""			r = item + '->' + items[item]""")
print("""			print(r)""")
print("""		m = sd.ReadNext()""")
m = sd.ReadNext()
while m != None:
    print(m.Formula())
    items = m.GetDataItems()
    for item in items.keys():
        r = item + "->" + items[item]
        print(r)
    m = sd.ReadNext()

print("""\nWrite StructureData into an SD file""")

print(""">>>sd = SDFileWriter.OpenFile('out.sdf', OverWrite)""")
print(""">>>m = StructureData.LoadData('CCC')""")
print(""">>>m.SetDataItem('atomcount', '3')""")
print(""">>>sd.WriteStructure(m)""")
print(""">>>m = StructureData.LoadData('C1CCCCC1')""")
print(""">>>m.SetDataItem('atomcount', '6')""")
print(""">>>sd.WriteStructure(m)""")
sd = SDFileWriter.OpenFile("out.sdf", OverWrite)
m = StructureData.LoadData("CCC")
m.SetDataItem("atomcount", "3")
sd.WriteStructure(m)
m = StructureData.LoadData("C1CCCCC1")
m.SetDataItem("atomcount", "6")
sd.WriteStructure(m)
print("""\nWelcome to ChemScript!""")
newInput("Press <Enter> to quit ...")
