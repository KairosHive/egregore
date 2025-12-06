# enhanced_miner.py — Advanced Archetype Mining with PDF, Folders, and LLM Refinement
"""
Enhanced corpus mining with:
  • Folder scanning for images (recursive)
  • PDF text extraction with smart chunking strategies
  • LLM-powered archetype derivation (Cloudflare or local)
  • Embedding clustering visualization
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Literal
from pathlib import Path
import numpy as np
import re
import json
import hashlib
from collections import Counter, defaultdict

# Optional imports with graceful fallbacks
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

DIRECTIONAL_ARCHETYPE_PROMPT = """You are a Semiotic Alchemist.

YOUR GOAL: 
Transmute raw concepts from a corpus into abstract Archetypes based on provided Filters (e.g., Directions, Seasons).

STRICT CONSTRAINTS:
1. ARCHETYPE NAMES: 
   - MUST be 1 or 2 words maximum.
   - NEVER start with "The".
   - MUST be abstract/symbolic (e.g., "OBLIVION", "DARK ZENITH", "SILENT VECTOR").
2. DESCRIPTORS: 
   - SINGLE WORDS ONLY. No phrases. No spaces.
   - MUST be complex, rare, or technical words (e.g., "petrichor", "isomorphic", "void").

THE PROCESS (Lateral Thinking):
1. Filter: "North" (Cold/Logic). Corpus: "Computer code, silence, glass".
2. Direct connection (Boring): "Frozen Code".
3. Lateral connection (Alchemical): "sarcophagus" or "cryogenize".

INSTRUCTIONS:
Synthesize the "Third Meaning" (Tertium Quid) between the Filter and the Corpus.

OUTPUT FORMAT (JSON):
{
  "archetypes": [
    {
      "name": "ARCHETYPE_NAME (No 'The')",
      "descriptors": ["word1", "word2", ... (Single words only)],
      "essence": "A philosophical justification of this remote connection."
    }
  ]
}
"""

def _load_cloudflare_credentials() -> Tuple[str, str]:
    """Load Cloudflare credentials from environment or secrets.toml."""
    import os
    
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    
    if not account_id or not api_token:
        try:
            try:
                import tomli
            except ImportError:
                import tomllib as tomli  # Python 3.11+ built-in
            secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                with open(secrets_path, "rb") as f:
                    secrets = tomli.load(f)
                cf = secrets.get("cloudflare", {})
                account_id = account_id or cf.get("account_id", "")
                api_token = api_token or cf.get("api_token", "")
        except Exception as e:
            print(f"[Cloudflare] Warning: Could not read secrets.toml: {e}")
    
    return account_id, api_token


# ============================================================
# Data structures
# ============================================================

@dataclass
class TextChunk:
    """A chunk of text with provenance metadata."""
    id: str
    text: str
    source: str  # filename or URL
    page: Optional[int] = None
    chunk_idx: int = 0
    strategy: str = "paragraph"  # paragraph, sentence, sliding, semantic
    meta: Dict = field(default_factory=dict)


@dataclass
class ImageItem:
    """An image with metadata."""
    id: str
    path: str
    source_folder: str
    filename: str
    description: Optional[str] = None  # LLM-generated description
    meta: Dict = field(default_factory=dict)


@dataclass 
class MinerCorpus:
    """Collection of text chunks and images for mining."""
    text_chunks: List[TextChunk] = field(default_factory=list)
    images: List[ImageItem] = field(default_factory=list)
    
    def __len__(self):
        return len(self.text_chunks) + len(self.images)
    
    def summary(self) -> Dict:
        return {
            "text_chunks": len(self.text_chunks),
            "images": len(self.images),
            "sources": list(set(c.source for c in self.text_chunks)),
        }


# ============================================================
# PDF Extraction
# ============================================================

class PDFExtractor:
    """Extract text from PDFs with multiple backend support."""
    
    @staticmethod
    def extract_text(pdf_path: str | Path, backend: str = "auto") -> List[Dict]:
        """
        Extract text from PDF, returning list of {page, text} dicts.
        
        Args:
            pdf_path: Path to PDF file
            backend: "pymupdf", "pdfplumber", or "auto"
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Auto-select backend
        if backend == "auto":
            if fitz is not None:
                backend = "pymupdf"
            elif pdfplumber is not None:
                backend = "pdfplumber"
            else:
                raise ImportError(
                    "No PDF backend available. Install: pip install pymupdf or pip install pdfplumber"
                )
        
        pages = []
        
        if backend == "pymupdf" and fitz is not None:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
            doc.close()
            
        elif backend == "pdfplumber" and pdfplumber is not None:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"page": i + 1, "text": text})
        else:
            raise ValueError(f"Backend '{backend}' not available")
        
        return pages


# ============================================================
# Text Chunking Strategies
# ============================================================

class TextChunker:
    """Smart text chunking with multiple strategies."""
    
    @staticmethod
    def chunk_paragraphs(
        text: str,
        min_length: int = 50,
        max_length: int = 1000,
        overlap: int = 0
    ) -> List[str]:
        """Split by double newlines (paragraphs), merge small ones."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Merge small paragraphs
        merged = []
        buffer = ""
        for p in paragraphs:
            if len(buffer) + len(p) < max_length:
                buffer = (buffer + "\n\n" + p).strip() if buffer else p
            else:
                if buffer and len(buffer) >= min_length:
                    merged.append(buffer)
                buffer = p
        if buffer and len(buffer) >= min_length:
            merged.append(buffer)
        
        return merged
    
    @staticmethod
    def chunk_sentences(
        text: str,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
        min_length: int = 50
    ) -> List[str]:
        """Split by sentences, group into chunks with optional overlap."""
        # Simple sentence splitting (handles common cases)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text] if len(text) >= min_length else []
        
        chunks = []
        i = 0
        while i < len(sentences):
            chunk_sents = sentences[i:i + sentences_per_chunk]
            chunk = " ".join(chunk_sents)
            if len(chunk) >= min_length:
                chunks.append(chunk)
            i += sentences_per_chunk - overlap_sentences
            if i <= 0:
                i = sentences_per_chunk  # prevent infinite loop
        
        return chunks
    
    @staticmethod
    def chunk_sliding_window(
        text: str,
        window_size: int = 512,
        stride: int = 256,
        min_length: int = 50
    ) -> List[str]:
        """Fixed-size sliding window (character-based)."""
        text = text.strip()
        if len(text) < min_length:
            return []
        
        chunks = []
        for start in range(0, len(text), stride):
            chunk = text[start:start + window_size].strip()
            if len(chunk) >= min_length:
                chunks.append(chunk)
            if start + window_size >= len(text):
                break
        
        return chunks
    
    @staticmethod
    def chunk_semantic(
        text: str,
        embedder: Callable[[List[str]], np.ndarray],
        similarity_threshold: float = 0.7,
        min_length: int = 50,
        max_length: int = 1500
    ) -> List[str]:
        """
        Semantic chunking: merge consecutive sentences while similar,
        break when similarity drops below threshold.
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) >= 20]
        
        if len(sentences) < 2:
            return [text] if len(text) >= min_length else []
        
        # Embed all sentences
        try:
            embs = embedder(sentences)
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        except Exception:
            # Fallback to paragraph chunking if embedding fails
            return TextChunker.chunk_paragraphs(text, min_length=min_length, max_length=max_length)
        
        chunks = []
        buffer = [sentences[0]]
        buffer_emb = embs[0]
        
        for i in range(1, len(sentences)):
            sim = float(np.dot(buffer_emb, embs[i]))
            combined_len = sum(len(s) for s in buffer) + len(sentences[i])
            
            if sim >= similarity_threshold and combined_len < max_length:
                buffer.append(sentences[i])
                # Update buffer embedding (running average)
                buffer_emb = (buffer_emb * (len(buffer) - 1) + embs[i]) / len(buffer)
                buffer_emb = buffer_emb / (np.linalg.norm(buffer_emb) + 1e-8)
            else:
                chunk = " ".join(buffer)
                if len(chunk) >= min_length:
                    chunks.append(chunk)
                buffer = [sentences[i]]
                buffer_emb = embs[i]
        
        # Final buffer
        chunk = " ".join(buffer)
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        return chunks


# ============================================================
# Folder Scanning
# ============================================================

class FolderScanner:
    """Scan folders for images and text files."""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
    TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".json"}
    
    @classmethod
    def scan_images(
        cls,
        folder: str | Path,
        recursive: bool = True,
        max_files: int = 1000
    ) -> List[ImageItem]:
        """Scan folder for image files."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        pattern = "**/*" if recursive else "*"
        images = []
        
        for path in folder.glob(pattern):
            if path.suffix.lower() in cls.IMAGE_EXTENSIONS and path.is_file():
                item = ImageItem(
                    id=hashlib.md5(str(path).encode()).hexdigest()[:12],
                    path=str(path),
                    source_folder=str(folder),
                    filename=path.name,
                    meta={"relative_path": str(path.relative_to(folder))}
                )
                images.append(item)
                
                if len(images) >= max_files:
                    break
        
        return images
    
    @classmethod
    def scan_pdfs(cls, folder: str | Path, recursive: bool = True) -> List[Path]:
        """Find all PDFs in a folder."""
        folder = Path(folder)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return list(folder.glob(pattern))


# ============================================================
# Semantic Word Matcher (Fast alternative to Vision LLM)
# ============================================================

# Default vocabulary - evocative words for archetype mining
DEFAULT_VOCABULARY = """
abstract abyss alchemy ancient angular archaic arcane ascending asymmetric 
atmospheric aurora austere awakening axis azure
balance baroque beacon becoming bifurcation blazing blossoming bold boundary 
bridge brilliant broken burning
calcified calm cascade celestial centered chaos chromatic cipher circular 
clarity clashing cold collapse combustion complexity concentric condensed 
conflicting convergence cosmic crackling crimson crossroads crystalline 
curious cyclical
dark dawn decay deconstruction deep dense descent desire diagonal diffuse 
dimensional discord dissolution distant divine dormant duality dusk dynamic
echo eclipse edge effervescent electric elemental elevated emanating ember 
emergent empty endless energy enigmatic ephemeral equilibrium eroded essence 
eternal ethereal evolution expanding explosive
fading falling fervent fierce fiery filament final flame flickering floating 
flow fluid flux focal folding forbidden forest forgotten formless fractured 
fragile fragment frozen fused fusion
gateway geometric ghost glacial gleaming glow golden gradient gravity growth 
guardian
hallowed harmony haze heavy helical hidden hieroglyphic hollow horizon 
hovering hybrid
ignited illuminated illusion immense immersive implosion incandescent 
incomplete infinite inner inscribed interlocking intersection intimate 
intricate inverted iridescent isolated
jagged journey junction juxtaposed
kinetic knot
labyrinth lacework layered leaping lens light liminal linear liquid lonely 
loop lost luminous lunar
magnetic majestic manifold matrix mechanical melancholic membrane memory 
merged metallic metamorphosis midnight minimal mirage mirror misty molten 
monolithic mosaic motion muted mysterious mystic
nascent natural nebulous nested network night noble nocturnal nomadic 
nostalgic nucleus
oblique obscured oceanic ominous oneiric opaque opening opposing orbit 
organic origin oscillating otherworldly outer overgrown
pale paradox parallel passage path pattern peaceful peak peeling pendulum 
peripheral perpetual phantom phosphorescent piercing pillar pinnacle planar 
plasma porous portal potential power primal pristine profound projection 
protective pulsing pure
quantum quartz quest quiet
radial radiant raw rebirth receding reciprocal recursive reflected refracted 
remnant renewal repetition resonance restless revelation rhythm ripple rising 
ritual rooted rotating rugged runic
sacred scattered scattered sealed seamless secret seed seeping serene shadow 
sharp shattered shell shimmering silent silhouette singular skeletal smooth 
soft solar solitary solid somber sonic sovereign space spectral sphere spiral 
split sprawling stable stacked stark static stellar stone storm stranded 
stratified stream strong structure submerged subtle suffused summit sunken 
superimposed surging suspended swirling symbolic symmetrical synchronous
tactile tangled temporal tender tension tessellated textured threshold 
timeless tonal topographic towering trace transcendent transformation 
translucent transmission transparent traversing trembling tribal twilight 
twisted
umbral unbound undulating unified unknown unraveling untamed unveiled 
uplifting upward urban
vacant vapor vast vector veiled velocity verdant vertical vessel vibrant 
visceral visible vital vivid void volcanic vortex
waning warm warp watcher wave weathered web weight whisper wild winding 
window wisdom withdrawn woven
zenith zero zonal
""".split()

# More abstract/symbolic words for deeper semantic matching
ARCHETYPE_VOCABULARY = """
abandonment absence absorption abstraction acceptance accumulation achievement 
activation adaptation admiration adventure affection affliction agency 
aggression agony allegiance ambiguity ambition ancestry anger anguish 
anticipation anxiety apathy apocalypse apparition appetite appreciation 
apprehension arcana archetype ascension aspiration assertion attachment 
attraction authority autonomy awakening awareness
balance banishment baptism barrier becoming belief belonging betrayal 
binding birth blessing blindness bliss bloodline bondage boundary 
breakthrough breath brotherhood burden burning
calling calm captivity catalyst catharsis celebration ceremony challenge 
change chaos character choice chrysalis clarity cleansing closure 
coalescence collapse collision communion communion community compassion 
completion complexity comprehension compulsion concealment concentration 
conception condemnation confession confluence confrontation connection 
consciousness consecration constellation contemplation continuity 
contradiction convergence conversion conviction corruption cosmos courage 
covenant creation creature crisis crossing crown crucible crystallization 
cultivation curse cycle
danger darkness dawn death decay deception decision dedication defiance 
deliverance delusion demise denial departure dependence descent desire 
despair destiny destruction determination devotion dichotomy dimension 
direction disappearance discovery disgrace disintegration displacement 
dissolution distance distortion divergence divination division dominion doom 
doorway doubt dragon dream duality dust duty dwelling
earth echo eclipse ecstasy edge ego element elevation emanation embodiment 
emergence emotion emperor empress emptiness enchantment encounter ending 
endurance energy enigma enlightenment ennui entity entropy ephemera epiphany 
equality equilibrium erosion eruption escape essence eternity ether 
evocation evolution exaltation excavation exchange exclusion execution 
exhaustion exile existence exodus expansion expectation experience 
exploration exposure expression extension extinction extraction
fabric facade failure faith fall falling fame family famine fate father fear 
feast fertility fever fiction fidelity field fire flight flood flow focus 
folly fool force foresight forest forgetfulness forgiveness form formation 
fortune foundation fracture fragment fragrance freedom frost fulfillment 
fury fusion future
garden gate gathering genesis ghost gift glacier glory gnosis goddess gold 
grace gradient grain gravity grief growth guardian guidance guilt
habit hallucination halo hammer hand harmony harvest haunting healing heart 
hearth heat heaven heir herald heritage hero hesitation hierarchy history 
hollow home honor hope horizon horror host house humanity humility hunger 
hunter hunting
icon ideal identity idol ignorance illumination illusion image imagination 
immanence immensity immortality impact impermanence implication imprisonment 
impulse incarnation inception inclusion incompleteness independence 
indication individuality indulgence inertia infancy infection infinity 
influence inheritance initiation innocence innovation inquiry inscription 
insight inspiration instinct institution instruction integration integrity 
intellect intensity intention interface interior interpretation interruption 
intersection intimacy introspection intuition invasion invention inversion 
investigation invocation iron irony island isolation iteration
journey joy judge judgment junction justice
keeper key kin kindred king kingdom kinship kiss knight knowledge
labyrinth ladder lamentation lamp land landmark landscape language lantern 
lapse latency law layer leadership leap legacy legend lens lesson liberation 
library life light lightning limit lineage link listener lodge logic 
loneliness longing lord loss love loyalty luminescence lure
machine madness magic magician maiden manifestation manipulation mantle 
marriage martyr mask mastery material matrix maturation maturity maze meaning 
measure mechanism meditation medium melancholy melody membrane memory mentor 
mercy merger message messenger metamorphosis metaphor method midnight 
migration milestone mind minister miracle mirror mission mist mixture model 
moment momentum monster moon mortality mother motion mountain mourning 
movement multiplicity muse music mystery mysticism myth
name narrative nation nature navigation nebula necessity nectar negation 
nemesis nest network nexus night nightmare nobility node nomad north 
nostalgia nothing nourishment novelty nucleus
oath obedience object obligation oblivion obscurity observation obsession 
obstacle occupation ocean odyssey offering offspring omen omniscience oneness 
opening operation opposition oracle orbit order ordination organ organism 
organization origin ornament orphan other outcome outer outlaw output outset 
outside outsider overcoming overflow overlord oversight overthrow
pact pain palace paradigm paradise paradox parallel parasite pardon parent 
pariah participation particle partition partner passage passion past path 
patience patriarch pattern pause peace peak pendulum penetration penance 
perception perdition perfection performance peril permanence permission 
perpetuity persecution perseverance persistence person persona perspective 
pestilence petition phantom phase phenomenon phoenix phrase pilgrimage pillar 
pinnacle pioneer pit place plague plan plane planet plant platform play 
pleasure pledge plenty plot plunge point poison polarity pole pool portal 
portion position possession possibility posterity posture potential poverty 
power prayer preacher precedent precipice precision predator prediction 
preference pregnancy prelude preparation presence preservation pressure 
pretense prey pride priest primal prime primordial prince princess principle 
priority prison privacy privilege probability probe problem procedure process 
proclamation prodigy production profanity profession profit profundity 
progenitor progress prohibition project proliferation promise proof prophecy 
prophet proportion proposal proposition protection prototype providence 
provision proximity prudence psyche pulse punishment purgatory purification 
purity purpose pursuit puzzle
quadrant quality quarantine queen quest question quiescence quiet quintessence
radiance radiation rage rain rainbow range rapture rarity ratio rationale 
raven ravine ray reaction reader reading realization realm reason rebellion 
rebirth receipt reception reciprocity reckoning recognition recollection 
reconciliation record recovery recreation redemption reduction reed 
reference reflection reformation refuge refusal regeneration region regret 
regulation reign rejection relation release relic relief religion remainder 
remembrance remnant removal renaissance renewal renunciation repair 
repetition replacement replica repose representation repression reproduction 
reputation requiem rescue research resemblance reservation reservoir 
residence resignation resilience resistance resolution resonance resource 
respect response responsibility rest restoration restraint restriction 
resurrection retention retribution return revelation revenge reverence 
reversal revival revolution reward rhythm riddle rift right rigidity ring 
ripple rise risk rite ritual river road rock role romance root rope rose 
rotation ruin rule rupture rush rust
sacrifice sadness safety saga sage sail saint salvation sanctity sanctuary 
sand saturation savage scale scar scatter scene scent scepter schism scholar 
science scope score scroll sea seal search season seat secrecy secret section 
security seed seeker seer selection self sensation sense sensitivity sentence 
sentinel separation sepulcher sequence seraph serenity serpent servant 
service session settlement seven shade shadow shaft shame shape shard shelter 
shepherd shield shift shimmer ship shock shore shrine shroud sibyl sick siege 
sight sigil sign signal significance silence silk silver similarity 
simplicity simulation sin sincerity single singularity sister site situation 
skeleton skill skull sky slave sleep slumber smoke snake snow society soil 
soldier solemnity solitude solution son song sorcerer sorrow soul sound 
source south sovereignty space span spark speaker specialization specimen 
spectacle spectrum speculation speech speed spell sphere spider spiral spirit 
splendor split spontaneity spring stability stack staff stage stagnation 
stain stake stance standard star stare state station statue status steam 
steel step stigma stillness stimulus stock stone storage storm story strain 
strand stranger strategy stratum stream strength stress stride strife strike 
string structure struggle student study stupor style subject submission 
subordination substance substitution subtlety succession suffering 
sufficiency suggestion suicide sum summit summons sun sunrise sunset 
sunshine superficiality superiority superstition supplement support 
suppression supremacy surface surge surprise surrender surveillance survival 
survivor suspension suspicion sustenance swamp swan swarm sway sweep 
sweetness swift sword symbol symmetry sympathy symptom synchronicity 
synergy synthesis system
table tablet taboo taint tale talent talisman tapestry target task taste 
teacher teaching tear technique technology temple temptation tendency tender 
tension terminus terrain territory terror testament testimony texture theorem 
theory thief thing thinker thinking thirst thistle thorn thought thread 
threat threshold throne thunder tide tiger time timelessness titan title 
token tolerance tomb tomorrow tone tongue tool tooth top torch torment 
tornado torrent torture touch tower trace track trade tradition tragedy 
trail train trait traitor trance transcendence transference transformation 
transgression transition translation transmission transmutation transparency 
transport trap trauma travel treasure treatment treaty tree trek tremor trend 
trial triangle tribe tribute trick trigger trilogy trinity trip triumph 
trouble truce trunk trust truth tunnel turbulence twilight twin type tyranny
ubiquity ugliness ultimate umbra umbrella uncertainty unconscious 
understanding undertaking undulation unification uniformity union uniqueness 
unity universal unknown unrest unveiling uprising uproar urgency usage 
utilization utopia utterance
vacancy vacuum vagabond vagueness vale validation validity valley valor value 
vampire vanity variation variety vault vector veil velocity vendetta 
veneration vengeance venture verdict verification verse version vertigo 
vessel vestige veteran vibration vice victim victory view vigilance vigor 
villain vine violence violet virgin virtue virus viscosity visibility vision 
visit visitor vista vitality vocation void volcano volume volunteer vortex 
vote vow voyage vulnerability vulture
wage wager waiting wake walk wall wanderer wandering war ward warfare warmth 
warning warrior waste watcher watchfulness water wave way weakness wealth 
weapon weather weave web wedding weed weight welcome well west wheel 
whirlpool whirlwind whisper white whole wholeness wickedness widow width wife 
wild wilderness will wind window wine wing winter wisdom wish wit witch 
withdrawal witness wizard woe wolf woman womb wonder wood word work world 
worm worry worship worth wound wrath wreath wreck wrestling writing wrong
yard yarn year yearning yellow yield youth
zeal zealot zenith zephyr zero zodiac zone
""".split()

# Animals - creatures from all habitats
ANIMAL_VOCABULARY = """
albatross alligator alpaca anaconda anchovy anemone angelfish ant anteater 
antelope ape armadillo asp baboon badger barracuda basilisk bass bat bear 
beaver beetle bison blackbird boa boar bobcat buffalo butterfly buzzard 
camel canary capybara cardinal caribou carp cat caterpillar catfish 
centipede chameleon cheetah chimpanzee chinchilla chipmunk cicada clam cobra 
cockatoo cockroach cod condor coral cougar cow coyote crab crane crawfish 
cricket crocodile crow cuckoo deer dingo doe dog dolphin donkey dove 
dragonfly duck eagle eel egret elephant elk emu falcon ferret finch firefly 
flamingo flea flounder fly fox frog gazelle gecko gerbil giraffe gnat gnu 
goat goldfish goose gorilla grasshopper grizzly grouper grouse gull hamster 
hare hawk hedgehog heron hippo hornet horse hound hummingbird hyena ibex 
ibis iguana impala jackal jackrabbit jaguar jay jellyfish kangaroo kestrel 
kingfisher kiwi koala koi komodo krill ladybug lamb lark lemming lemur 
leopard limpet lion lizard llama lobster locust loon lynx macaw mackerel 
magpie manatee mandrill manta mantis marlin marmot marten mastiff mayfly 
meerkat mink mockingbird mole mollusk mongoose monkey moose mosquito moth 
mouse mule mussel narwhal nautilus newt nightingale ocelot octopus opossum 
orangutan orca oriole osprey ostrich otter owl ox oyster panda panther 
parakeet parrot partridge peacock pelican penguin perch pheasant pig pigeon 
pike piranha platypus plover pony porcupine porpoise possum prawn puma 
python quail rabbit raccoon ram rat rattlesnake raven ray reindeer rhino 
roadrunner robin rooster salamander salmon sardine scorpion seagull seahorse 
seal shark sheep shrew shrimp silkworm skate skunk sloth slug snail snake 
snapper sparrow spider squid squirrel stallion starfish stingray stork 
sturgeon swallow swordfish tapir tarantula tarsier termite tern thrush tick 
tiger toad tortoise toucan trout tuna turkey turtle urchin viper vulture 
wallaby walrus wasp weasel whale wildcat wildebeest wolverine woodpecker 
wren yak zebra
""".split()

# Body parts and anatomy
BODY_VOCABULARY = """
abdomen ankle antler aorta appendage arch arm artery backbone belly bicep 
bladder blood bone bowel brain breast brow buttock calf cartilage cell cheek 
chest chin claw coccyx collarbone cornea cortex cranium crown dermis digit 
ear elbow embryo entrails epidermis esophagus eye eyebrow eyelash eyelid 
face fang feather femur fiber fin finger fingernail fist flesh foot forearm 
forehead fur gall gills gland gonad groin gullet gum gut hair hamstring hand 
haunch head heart heel hip hoof horn humerus hump intestine iris jaw joint 
jugular kidney knee knuckle larynx leg lid limb lip liver lobe loin lung 
mane mandible marrow maw membrane midriff molar mouth mucus muscle nape 
navel neck nerve nipple nose nostril organ ovary palm pancreas paw pelvis 
phallus pore pupil radius rectum retina rib rump scalp scapula scar shoulder 
sinew skeleton skin skull snout sole spinal spine spleen sternum stomach 
strand stubble stump synapse tail talons teeth temple tendon testicle thigh 
thorax throat thumb thyroid tibia tissue toe tongue tonsil tooth torso 
trachea trunk tusk udder ulna umbilical uterus uvula vein ventricle vertebra 
vessel viscera waist whisker windpipe wing womb wrist
""".split()

# Textures and materials
TEXTURE_VOCABULARY = """
abrasive absorbent angular ashen barbed beaded beady beveled blistered 
bloated blotchy braided bristly brittle bubbly bumpy burnished bushy buttery 
calcified callused chalky chapped charred checkered chipped chunky clammy 
clayey clear clogged cloudy clumpy coagulated coarse cobbled compacted 
congealed corded corduroyed corky corroded corrugated cottony cracked 
crackled cratered crazed creased creamy creased creped crimped crinkled 
crisp crumbly crushed crusty crystalline curdled curly cushioned damp 
dehydrated delicate dented dimpled dirty downy drenched dried dripping dry 
dull dusty elastic embossed enameled engraved eroded faceted faded feathered 
feathery felted fermented fibrous filmy fine flagstone flaky flat flaxen 
fleshy flexible flinty floppy floury fluffy fluted foamy foliated foliage 
frayed frazzled fretted fringed frothy frosted frozen furry furrowed fuzzy 
gauzy gelatinous gilded glistening gloopy glossy gluey glutinous gnarled 
gnarly goopy grainy granular granulated grated greasy gritty grooved grubby 
gummy hairy hardened hazy honeycombed horny icy jagged jellied jelly knobbly 
knotted knotty lacy lamellar laminated latticed leaden leafy leathery lichened 
limp linen lintlike liquefied lissome lithic loamy lofty loose lopsided 
lumpy lustrous luxuriant macular marbled marshy matted mealy meaty membranous 
meshy metallic milky mineral miry misshapen moist moldy mossy mottled muddy 
muddled murky mushy musty napped nappy netted nubby numbed oily oozy ossified 
oxidized packed padded parchment pasty patchy patterned pearly pebbled 
peeling pelted perforated petal petrous pillowy pimpled pitted plaited 
plastic pleated plush pocked pockmarked polished porous potholed powdered 
powdery prickly puckered pudgy puffy pulpy punctured quilted ragged rasping 
ratty raveled raw reedy resinous ribbed ridged rigid rimed rippled roiled 
ropy rotten rough rounded rubbery ruffled rugged rumpled runny russet rusted 
rustic rutted sanded sandy satiny scabbed scabrous scalloped scaly scarred 
scorched scraggly scraped scratched scratchy scruffy scummy scurfy seamed 
seared seedy serrated shaggy shagreen shattered sheer shelled shiny shirred 
shivered shredded shriveled shrubby silken silky silt silty silvered sinuous 
skeletal skinned slatted sleek slick slimed slimy slippery sloppy sloshy 
sludgy slushy smeared smoggy smoky smooth smudged snaggy snarled soaked 
soapy sodden soft soggy soiled solid sooty sore soupy speckled spherical 
spiked spiky spiny splotchy spongy spotted sprigged springy spun squashy 
squelchy stained stalky starched starchy stiff stippled stitched stodgy 
stony stranded strawy streaked streaky striated striped stubbled stubbly 
studded stuffed stung stuffy stumpy succulent sueded sugary supple swampy 
sweaty swollen tacky tangled tarnished tarry tattered tender tepid terraced 
textured thick thickened thorny threadbare threaded thatched tiled tinselly 
tired toasted toned tongued tooled toothed torn toughened trampled treacly 
tremulous trickling trimmed tubular tufted tumid turbid turgid tweed twiggy 
twilled twisted uneven unfinished unpolished upholstered varnished vascular 
veined velvety venous verrucose viscid viscous vitreous vulcanized wadded 
warped watered watery waved wavy waxen waxy weathered weedy welted wettened 
whiskered wiggly wiry withered wizened wondrous wooded woolen woolly worn 
woven wrinkled yeasty
""".split()

# Sensations and feelings
SENSATION_VOCABULARY = """
ache acrid acute agony alert alive alluring anguished anxious aroused 
asleep astringent atingle awake balmy bitter blazing blinding blissful 
bloated boiling bracing breathless breezy brisk bruised burning buzzing 
calm captivated caressed chafed charmed chilled chilling clammy cleansed 
coarse cold comforted comfortable comforting constricted cool cooling 
cramped crawling crisp crushing cutting damp dazed deadened deafening 
delicate delightful dense desiccated dizzy drained dreamy drowsy dry dull 
ecstatic effervescent electric electrified elated energized engorged enraged 
euphoric exasperated excited exhausted exhilarated famished fatigued feeble 
fervent feverish fiery firm floating flushed freezing fresh frigid 
fulfilled fuzzy giddy glacial glaring glowing gnawing grating greasy grimy 
gritty grounded harsh healing heated heavenly heavy hollow hot humid hungry 
hushed hypnotic icy immersed inflamed invigorated inviting irritated itching 
itchy jarring jittery jolted keen kinetic languid lethargic light lightheaded 
limp lively livid longing lulled luscious maddening magnetic mellow melting 
moist muggy muted nauseated nauseous nettled nibbling nipping numb numbing 
oppressive overheated overwhelmed painful palpitating parched peaceful 
penetrating peppery perfumed pining pins-and-needles pleasurable pounding 
pressing prickling prickly pulsating pungent quaking quenched quivering 
radiant raging rasping raw reeling refreshed refreshing relaxed relaxing 
relieved repulsive restless restoring revived rhythmic rigid rippling 
roaring rough rumbling rushing rustic salty sated satiated saturated savorly 
scalded scalding scathing scorched scorching scratchy searing sensual serene 
sharp shattered shivering shocking silky sizzling sleepy slick slippery 
sloppy slow sluggish smarting smoldering smooth smothered smothering snug 
soaked sodden soft soothed soothing sore sorrowful sparkling spent spicy 
splitting spongy stabbing stale starving steaming steamy sticky stiff 
stifled stifling stinging stirred strained strangled streaming stressed 
stretched stroked stroking strong stuffy stunned stupefied subtle suffocated 
suffocating sultry sumptuous supple surging swaying sweaty swelling 
sweltering swift swooning tart taut tender tense tepid terrible terrified 
thirsty thorny throbbing thrumming thunderous tickling ticklish tight 
tingling tired torpid touched touching tranquil trembling tremulous 
twitching uncomfortable uneasy unnerved unwinding velvety vibrating vibrant 
violent vitalized warm warped washed wasted watery weak weighted weighty 
welcoming wet whipped wild winded windswept wobbly worn wounded wrenching 
wretched writhing yearning zapped zesty zingy
""".split()

# Tastes and flavors
FLAVOR_VOCABULARY = """
acerbic acidic acrid aftertaste alkaline ambrosial appetizing aromatic 
astringent balsamic biting bitter bland blistering bracing brackish briny 
burnt buttery candied caramelized caustic charred cheesy chemical chocolatey 
cinnamon citrus cloying coconut corky crisp crunchy delectable delicate 
delicious divine doughy dry earthy eggy fermented fiery fishy flavorful 
floral fragrant fresh fruity fudgy funky gamey garlicky gingery gooey 
grassy gravy greasy grilled gritty harsh hearty herbaceous herbal honeyed 
hoppy hot inedible insipid intense juicy lemony licorice lightly light 
luscious malty medicinal mellow metallic mild milky mineral minty moldy 
morish musky musty nasty nutmeg nutty oaky oily oniony organic overripe 
palatable peppery perfumed pickled piquant plain plastic pleasant plummy 
poignant puckering pulpy pungent putrid rancid raw refreshing resinous rich 
ripe roasted robust rotting saccharine saline sapid salty saturated saucy 
savory scorched sea seared seasoned sharp silky skunky smoky smooth soapy 
soft soothing sour sparkling spiced spicy spoiled stale starchy steamy 
stewed sticky stinging stringy strong succulent sugary sulfurous sweet 
syrupy tangy tart tasteless tasty tepid tinny toasted toasty tongue-numbing 
toxic treacly umami unappetizing unpalatable unsalted unsavory vanilla vegetal 
velvety vinegary virulent watered-down watery weak wild woody yeasty zesty 
zippy
""".split()

# Colors and hues
COLOR_VOCABULARY = """
alabaster amber amethyst apricot aqua aquamarine arctic ash auburn azure 
beige beryl bisque black blonde blood blush bone bordeaux brass brick 
bronze brown buff burgundy burnt butterscotch cadet camel camouflage canary 
candy caramel cardinal carmine carnation celadon celeste cerise cerulean 
chalk champagne charcoal chartreuse cherry chestnut chocolate chrome cider 
cinnabar cinnamon citrine citron claret clay coal cobalt cocoa coffee 
cognac copper coral cordovan corn cornflower cornsilk cosmic cranberry 
cream crimson crystal cyan daffodil damask dandelion dapple dark dawn 
denim desert dim drab dusk dusty ebony ecru eggplant eggshell electric 
emerald faded fawn fern fire flame flamingo flax flesh flora fluorescent 
forest fuchsia garnet ginger glacier glaucous gold golden granite grape 
graphite grass gray green grey gunmetal hazel heather heliotrope hemp 
henna hickory honey honeydew hot hunter hyacinth ice indigo iris iron 
ivory jade jasmine jasper jet juniper kelly khaki lapis lavender lemon 
light lilac lime linen magenta mahogany maize malachite mandarin mango 
maple maroon mauve melon mercury midnight mint mocha molten moss mulberry 
mustard navy neon neutral night noir nude oat ocher ochre olive onyx 
opal orange orchid oxford oyster pale paprika peach peacock pear pearl 
periwinkle persimmon pewter pine pink pistachio pitch platinum plum powder 
primrose prune puce pumpkin purple quartz raspberry raven red redwood 
rose rosewood rosy rouge royal ruby russet rust sable saffron sage salmon 
sand sangria sapphire scarlet seashell sepia shadow shamrock shell sienna 
silver sky slate sleet smoke snow sorrel steel stone straw strawberry 
sunflower sunny sunset tan tangerine taupe tawny teal terracotta thistle 
thyme tiger timber toast tobacco tomato topaz tropical turquoise ultramarine 
umber umber vanilla verdigris vermillion violet viridian walnut watermelon 
wheat white wine wintergreen wisteria wood xanthic yellow zinc
""".split()

# Natural elements and phenomena
NATURE_VOCABULARY = """
afterglow algae altitude amber ambergris archipelago asteroid atmosphere 
aurora avalanche badlands bank basin bay beach bedrock berg biome blizzard 
bloom bog boulder branch breeze brine brook butte caldera canyon cape 
cascade cataract cavern chasm cirque clay clearing cliff cloud coast 
coastline comet confluence continent coral cove crater creek crest crevasse 
crevice current cyclone dale dam dawn delta desert dew divide dome downpour 
driftwood drizzle drought dune dusk dust earthquake echo eddy embankment 
ember equator erosion escarpment estuary evergreen extinction falls fault 
fauna fen fern field fjord flood flora flurry foam fog foothill forest 
formation fossil fountain freshwater frond frost fungi gale galaxy garden 
geyser glacier glade glen gorge granite grassland gravel grove gulf gust 
hail harbor haven heath haze headland headwater highland hill hillside 
horizon hot-spring hurricane ice iceberg icecap inlet island isthmus 
jungle kelp knoll lagoon lake landmass landslide lava lawn lea ledge 
lightning loam lowland magma mangrove marsh marshland meadow meander mesa 
mist monsoon moon moonbeam moonlight moraine morass moss mound mountain 
mountaintop mud mudflat muskeg nebula oasis ocean outcrop overhang oxbow 
pass pasture peak peat peninsula permafrost piedmont pine plain planet 
plateau playa plume pond pool prairie precipice promontory puddle quarry 
quartz quicksand rain rainbow rainforest range rapids ravine reef ridge 
rift riparian river riverbed rivulet rock rockfall rubble runoff sand 
sandbar sandstone savanna scarp scree scrub sea seafloor seam seascape 
seashore season sediment shade shelf shoal shore shoreline shrub sierra 
silt sinkhole sky sleet slope slush smog smoke snow snowcap snowdrift 
snowfall snowfield snowflake snowmelt snowpack soil solstice sound source 
spring sprout squall stalactite stalagmite star steam steppe stone strait 
strand stratum stream streambed stump summit sun sunbeam sunlight sunrise 
sunset sunshine surf surge swale swamp swell taiga tarn tempest terrace 
thaw thicket thunder thunderhead tide tideland tidepool timber timberline 
tor tornado torrent tract tree trench tributary trough tsunami tundra 
undergrowth undertow upland updraft vale valley vapor vegetation veld 
volcano wake waterfall watershed waterspout wave wetland whirlpool wild 
wildfire wilderness wind windstorm woodland woods wrack
""".split()

# Emotions and mental states  
EMOTION_VOCABULARY = """
abhorrence admiration adoration affection agitation alarm alienation 
amazement ambivalence amusement anger annoyance anticipation antipathy 
anxiety apathy apprehension ardor arousal astonishment attraction aversion 
awe bewilderment bitterness bliss boredom calm certainty chagrin cheerfulness 
clarity closeness coldness comfort compassion complacency compunction concern 
confidence confusion consternation contempt contentment courage cowardice 
craving curiosity cynicism dazzlement defeat defiance dejection delight 
denial depression derangement desire desolation despair despondency 
detachment determination devastation devotion disappointment disbelief 
discomfort discontent disdain disgrace disgust disillusionment dismay 
disorientation displeasure dissatisfaction distraction distress distrust 
doubt dread eagerness earnestness ease ecstasy edginess elation 
embarrassment empathy emptiness enchantment encouragement enjoyment ennui 
enthusiasm envy equanimity euphoria exasperation excitement exhaustion 
exhilaration expectancy exuberance faith fascination fatigue fear ferocity 
fervency fervor fondness foreboding frenzy fright frustration fulfillment 
fury gaiety gladness gloom gloating glumness gratification gratitude greed 
grief grudge guilt gusto happiness hate hatred helplessness hesitancy 
homesickness hope hopelessness horror hostility humiliation hunger hurt 
hysteria idolatry impatience incredulity indifference indignation infatuation 
insecurity inspiration intimidation intrigue irritation isolation jealousy 
jitters jollity joy jubilation kindness lethargy levity liberation 
liveliness loathing loneliness longing love lust malaise malevolence 
malice marvel melancholy mellowness menace merriment mirth misery mistrust 
modesty mortification motivation mourning neediness negativity nervousness 
nihilism nostalgia nothingness numbness obsession offense optimism outrage 
overwhelm panic paranoia passion pathos patience peacefulness pensiveness 
perplexity perturbation pessimism piety pity pleasure possessiveness 
powerlessness pride puzzlement rage rapture reassurance regret rejection 
relaxation relief reluctance remorse repentance repulsion resentment 
resignation resolve restlessness reverence revulsion romance ruthlessness 
sadness satisfaction scorn security self-pity self-satisfaction sensation 
sensitivity sentimentality serenity shame shock shyness skepticism smugness 
somberness sorrow spite stagnation stillness stress stubbornness stupefaction 
submission suffering sullenness surprise suspense suspicion sympathy 
temptation tenderness tension terror thankfulness thrill timidity tolerance 
torment tranquility tribulation triumph troubledness trust turbulence 
uncertainty unease unhappiness upset urgency valor vanity vengeance vexation 
vigilance vulnerability wanderlust wariness warmth weariness whimsy 
wistfulness withdrawal woe wonder worry wrath yearning zeal zest
""".split()

# Sounds and acoustics
SOUND_VOCABULARY = """
babble bang bark bawl bay beat bellow blare blast bleat boom bray bubble 
bump burble burp buzz cackle call caterwaul chant chatter cheep chime 
chink chirp chirrup clack clang clank clap clash clatter click clink clip 
clop cluck clunk cooing crackle crash creak croak crow crump crunch cry 
din ding discord drone drone droning drum echo fizz flutter gasp giggle 
gnash grate grinding groan growl grumble grunt gurgle hiss honk hoot howl 
hum hubbub jangle jingle keen knock lilt lowing mewl moan muffle mumble 
murmur mutter neigh noise outcry patter peal peep ping pipe plonk plop 
plunk pop pounding prattle purr quack racket rap rasp rattle resonance 
reverberation ring ripple roar roll rumble rustle scream screech shriek 
shuffle sigh silence sing siren sizzle slam slap slosh slurp smack snap 
snarl sniff sniffle snore snort sob song splash splatter splutter sputter 
squall squawk squeak squeal squelch stammer static stridor strum swish 
swoosh thrum thud thump thunder tick tinkle toll tone toot trill trumpet 
tumult twang tweet twitter ululation uproar vibration wail warble whack 
wheeze whimper whine whinny whisper whistle whoop whir whoosh yelp yodel 
yowl zap zip zoom
""".split()

# Actions and movements
ACTION_VOCABULARY = """
absorb accelerate accumulate achieve activate adapt adhere adjust advance 
agitate align alter amplify anchor animate approach arch arise arrange 
ascend assemble assert attract awaken balance bathe bend bind blast blend 
blink bloom blow bob bolt bounce bow brace branch break breathe brew bridge 
bristle broadcast brush buckle bud build bulge bundle burn burst bury 
caress carry carve cascade cast catch center chain change channel charge 
chase churn circle clamp clasp claw cleanse clear cleave cling close 
cluster coalesce coast coil collapse collect collide combine combust 
compact compel compress conceal concentrate condense conduct confine 
congregate connect consume contain continue contract converge convey convulse 
coil corrupt couple cover crack crackle crash crawl create creep crisp 
crook cross crouch crumble crumple crush crystallize curl curve cushion 
cut dance dart dash decay deepen deflect deform delay deliver demolish 
depart deposit descend desert detach devour diffuse dig dilate dim dip 
direct discharge disconnect disperse dispel display dissolve distend 
distort distribute dive diverge divert divide dock dodge dominate drag 
drain drape draw dredge drench drift drill drink drip drive droop drop 
drown dull dump duplicate dwell dwarf echo edge eject elaborate elude 
emanate embark embed embrace emerge emit empty encircle enclose encompass 
engulf enlarge enter entomb entwine envelop erase erode erupt escape 
evade evaporate evolve excavate excite exclude execute exhale exhaust 
expand expel explode explore expose extend extract exude fade fall fan 
fasten feed ferment fill filter find fire fix flame flap flare flash 
flatten flee flex flick flinch fling flip flit float flock flood flow 
flourish fluctuate flush flutter fly foam focus fold follow force forge 
form formulate fracture fragment freeze fret fry fume funnel furl fuse 
gain gallop gape garner gasp gather generate germinate gesture gild give 
glance glare gleam glide glimmer glint glisten glitter glow gnaw gouge 
govern grab graft grasp grate graze greet grind grip grope grovel grow 
guide gulp gush gyrate halt hammer handle hang harden harness harvest 
hasten haul heal heap heave hide hinder hinge hit hoard hold hollow hook 
hop host hover hug hurl hurtle ignite illuminate immerse impale implode 
imprint incise incline include increase indent induce infect infiltrate 
inflate inflict inhabit inhale inject insert inspect inspire install 
integrate intensify interlock intertwine interweave introduce invade invert 
investigate isolate jab jangle jar jerk jet jiggle jolt jostle journey 
juggle jump kick kindle kiss knead kneel knit knock knot lace land lap 
lash latch launch layer lead leak lean leap leave level lever liberate 
lick lift light lighten linger link liquefy list listen litter live load 
loaf lodge loop loose loosen lower lug lull lumber lunge lurch lurk magnify 
make manifest manipulate march mark mash mass massage mature meander measure 
meet meld melt merge mesh migrate mingle mirror mist mix mobilize mold 
mount move multiply muffle murmur mushroom mutate navigate nest nestle 
nibble nod notch nourish nudge nurture observe occupy occur offend offer 
open operate orbit organize oscillate ooze overcome overflow overhang 
overlap overrun overtake overturn overwhelm pace pack paddle paint pair 
palpitate pan parade paralyze part partition pass paste pat patch patrol 
pause peck peel peep peer penetrate perch percolate permeate persist 
pervade pierce pile pinch pitch place plant plaster pleat pledge plod 
plop plot pluck plug plummet plunder plunge ply pocket point poise polish 
pollinate ponder pool pop portion pose position possess pound pour power 
practice praise prance precede precipitate preoccupy prepare press prevail 
prevent prick prime probe proceed process procure prod produce progress 
project proliferate prolong prompt propel protect protrude provide prowl 
pry puff pull pulp pulsate pulse pump punch puncture purge purify pursue 
push quake quarry quash quaver quell quest quiver radiate raid rain raise 
rake rally ramble rampage range rankle rant rattle ravage ravel reach 
reap rear rebound recede receive reclaim recline recognize recoil recover 
recreate recur redden reduce reel reflect refract refresh regain regenerate 
register reign reinforce reject rejoin rejoice relapse relax release 
relent relieve relinquish remain remake remember remove render renew 
repair repel replace replenish repose represent repress reproduce repulse 
request require rescue research reside resist resolve resonate respond 
rest restore restrain retain retard retract retreat retrieve return reveal 
reverberate reverse review revise revive revolve reward rid riddle ride 
rifle rig rim ring rinse ripple rise risk rival roam roar rock roll romp 
root rotate rouse route rove rub ruffle ruin rumble rummage run rupture 
rush rust sack sacrifice sail salvage sample sanctify sand sap saturate 
save savor saw scale scan scar scatter scavenge scoop scorch scour scout 
scowl scramble scrape scratch scrawl scream screen scribble scrub scrutinize 
scuffle sculpt scurry seal search sear season seat seclude secure seduce 
seep seethe seize select send sense separate sequester serve set settle 
sever shade shadow shake shatter shear shed shelter shield shift shimmer 
shine shiver shock shoot shorten shove show shower shred shriek shrink 
shrivel shroud shrug shudder shuffle shun shut shuttle sift sigh signal 
simmer sing sink sip siphon sit situate skate sketch skid skim skip skirt 
skulk slam slant slap slash slate slaughter sleep slice slide sling slink 
slip slit slither slope slosh slouch slow sluice slump smack smash smear 
smell smelt smile smirk smoke smolder smooth smother snap snare snarl 
snatch sneak sniff snip snoop snore snort snow snub snuff snuggle soak 
soar sob soften soil soldier solidify solve somersault soothe sort sound 
sow space span spark sparkle spatter spawn speak spear specialize speck 
speed spell spend spew spice spill spin spiral spit splash splatter splice 
splinter split spoil sponge spoon sport spot spout sprawl spray spread 
spring sprinkle sprint sprout spurt sputter squander square squash squat 
squawk squeeze squelch squint squirm stab stabilize stack stagger stain 
stake stall stamp stand staple stare start startle starve stash station 
stay steady steal steam steer stem step stew stick stiffen stifle still 
stimulate sting stink stipple stir stitch stock stomp stoop stop store 
storm straddle strafe strain strand strangle strap stray streak stream 
strengthen stress stretch strew stride strike string strip strive stroke 
stroll strut structure struggle strut study stuff stumble stump stun 
stupefy stutter subdue submerge submit subside substitute subtract succeed 
succumb suck suffer suffocate suggest suit summon sunbathe superimpose 
supervise supplement supply support suppress surface surge surmount surpass 
surround survey survive suspend sustain swab swaddle swagger swallow swamp 
swap swarm sway swear sweat sweep swell swerve swift swim swing swipe swirl 
switch swoop symbolize
tackle tag tail tailor take talk tame tamp tamper tangle tap taper target 
tarnish taste taunt tear tease telegraph telescope temper tempt tend tense 
terminate terrify test tether thaw thicken thin think thrash thread 
threaten thresh thrive throb throttle throw thrust thud thump thunder 
thwart tickle tie tighten tilt tingle tinkle tip tiptoe tire toast toddle 
toil toll tone toot topple torment torpedo toss total totter touch tour 
tousle tow tower trace track trade trail train trample transcend transfer 
transform transgress translate transmit transport trap travel traverse 
trawl tread treat trek tremble trend trespass trick trickle trigger trim 
trip triumph trod troop trot trouble trudge trumpet trundle trust try tuck 
tug tumble tunnel turn turtle tweak twine twinkle twirl twist twitch 
twitter type
unbind unbutton unchain uncloak uncoil uncork uncover uncurl undercut 
undergo underline undermine understand undertake undo undress undulate 
unearth unfasten unfold unfurl unhinge unite unknot unlace unlatch unleash 
unload unlock unmask unpack unpick unplug unravel unroll unseal unsettle 
untangle untie unveil unwind unwrap upend upgrade uphold uplift uproot 
upset upstage upturn urge use usher usurp utter
vacate vacuum validate vanish vanquish vaporize vault veer veil vent 
venture verify vex vibrate vie view vindicate violate visit visualize 
vitalize vocalize voice void volley volunteer vomit vote vouch vow voyage
wade waft wager waggle wail wait wake walk wallow wander wane want warble 
ward warm warn warp warrant wash waste watch water wave waver wax weaken 
wean wear weary weather weave wedge weed weep weigh welcome weld well 
wet whack whale wham wheel whet whimper whine whinny whip whir whirl 
whisk whisper whistle whittle whiz whoop widen wield wiggle wilt wince 
wind wink winnow wipe wish withdraw wither witness wobble wolf wonder 
woo work worm worry worship wound wrap wreak wreath wreathe wreck wrench 
wrest wrestle wriggle wring wrinkle write writhe
yank yap yawn yearn yell yelp yield yodel yowl
zap zero zigzag zip zone zoom
""".split()


class SemanticWordMatcher:
    """
    Fast image-to-words matching using CLIP embeddings and a vocabulary database.
    Much faster than Vision LLM - embeds image once, compares against pre-computed word embeddings.
    Uses FAISS for fast approximate nearest neighbor search when available.
    """
    
    def __init__(
        self,
        vocabulary: Optional[List[str]] = None,
        clip_model: str = "ViT-B/32",
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_faiss: bool = True,
    ):
        """
        Args:
            vocabulary: List of words to use. If None, uses all vocabulary lists (~4000 words)
            clip_model: CLIP model to use
            device: Device for computation
            cache_path: Path to cache word embeddings
            use_faiss: Use FAISS for fast similarity search (recommended)
        """
        import torch
        import hashlib
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Combine all vocabularies for rich semantic matching
        self.vocabulary = vocabulary or (
            DEFAULT_VOCABULARY + ARCHETYPE_VOCABULARY + 
            ANIMAL_VOCABULARY + BODY_VOCABULARY + 
            TEXTURE_VOCABULARY + SENSATION_VOCABULARY + 
            FLAVOR_VOCABULARY + COLOR_VOCABULARY + 
            NATURE_VOCABULARY + EMOTION_VOCABULARY + 
            SOUND_VOCABULARY + ACTION_VOCABULARY
        )
        self.vocabulary = sorted(set(self.vocabulary))  # Remove duplicates and sort for consistent caching
        self.clip_model_name = clip_model
        # Default to static folder so cache can be committed to repo
        self.cache_path = cache_path or str(Path(__file__).parent / "static" / "word_embeddings.npy")
        self.use_faiss = use_faiss
        
        # Ensure cache directory exists
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._clip_model = None
        self._clip_preprocess = None
        self._word_embeddings = None
        self._faiss_index = None
        self._word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        
        # Create a STABLE hash of vocabulary for cache validation (hashlib, not Python hash())
        vocab_str = "|".join(self.vocabulary) + "|" + clip_model
        self._vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()
        
    def _ensure_loaded(self):
        """Lazy load CLIP model and word embeddings."""
        # Check if already fully loaded
        if self._clip_model is not None and self._word_embeddings is not None:
            return
            
        try:
            import clip
            import torch
            
            # Try to load cached word embeddings FIRST (before loading CLIP model)
            cache_loaded = False
            if self._word_embeddings is None and self.cache_path and Path(self.cache_path).exists():
                try:
                    cache = np.load(self.cache_path, allow_pickle=True).item()
                    # Use vocab_hash for faster comparison
                    cached_hash = cache.get("vocab_hash")
                    if cached_hash == self._vocab_hash and cache.get("model") == self.clip_model_name:
                        self._word_embeddings = cache["embeddings"]
                        cache_loaded = True
                        print(f"[SemanticMatcher] Loaded cached word embeddings ({len(self.vocabulary)} words)", flush=True)
                    elif cache.get("model") != self.clip_model_name:
                        print(f"[SemanticMatcher] Cache model mismatch, will recompute", flush=True)
                    else:
                        print(f"[SemanticMatcher] Cache vocabulary changed, will recompute", flush=True)
                except Exception as e:
                    print(f"[SemanticMatcher] Cache load failed: {e}", flush=True)
            
            # Load the CLIP model if not already loaded
            if self._clip_model is None:
                print(f"[SemanticMatcher] Loading CLIP model {self.clip_model_name}...", flush=True)
                self._clip_model, self._clip_preprocess = clip.load(self.clip_model_name, device=self.device)
            
            if not cache_loaded and self._word_embeddings is None:
                # Compute word embeddings - batch to avoid memory issues
                print(f"[SemanticMatcher] Computing embeddings for {len(self.vocabulary)} words (one-time)...", flush=True)
                
                # CLIP can only tokenize so many words at once, batch them
                batch_size = 500
                all_features = []
                
                with torch.no_grad():
                    for i in range(0, len(self.vocabulary), batch_size):
                        batch = self.vocabulary[i:i+batch_size]
                        text_tokens = clip.tokenize(batch, truncate=True).to(self.device)
                        text_features = self._clip_model.encode_text(text_tokens)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        all_features.append(text_features.cpu())
                
                self._word_embeddings = torch.cat(all_features, dim=0).numpy()
                print(f"[SemanticMatcher] Computed {self._word_embeddings.shape[0]} word embeddings", flush=True)
                
                # Cache for next time
                try:
                    np.save(self.cache_path, {
                        "vocab_hash": self._vocab_hash,
                        "model": self.clip_model_name,
                        "embeddings": self._word_embeddings,
                        "vocab_size": len(self.vocabulary)
                    })
                    print(f"[SemanticMatcher] Cached word embeddings to {self.cache_path}", flush=True)
                except Exception as e:
                    print(f"[SemanticMatcher] Failed to cache: {e}", flush=True)
            
            # Build FAISS index for fast similarity search
            if self.use_faiss and self._faiss_index is None and self._word_embeddings is not None:
                self._build_faiss_index()
            
            # Final check - ensure embeddings are loaded
            if self._word_embeddings is None:
                raise RuntimeError("Failed to load or compute word embeddings")
                        
        except ImportError as e:
            raise ImportError(f"CLIP is required for SemanticWordMatcher. Install with: pip install git+https://github.com/openai/CLIP.git. Error: {e}")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search."""
        try:
            import faiss
            
            # Normalize embeddings for cosine similarity (use inner product on normalized vectors)
            embeddings = self._word_embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            
            # Get embedding dimension
            d = embeddings.shape[1]
            
            # For ~4000 words, a flat index is fine and exact
            # For larger vocabularies (10k+), use IVF index
            if len(self.vocabulary) > 10000:
                # IVF index for large vocabularies
                nlist = min(100, len(self.vocabulary) // 40)  # Number of clusters
                quantizer = faiss.IndexFlatIP(d)
                self._faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                self._faiss_index.train(embeddings)
                self._faiss_index.add(embeddings)
                self._faiss_index.nprobe = 10  # Search 10 clusters
                print(f"[SemanticMatcher] Built FAISS IVF index ({len(self.vocabulary)} words, {nlist} clusters)", flush=True)
            else:
                # Flat index for smaller vocabularies (exact search, still very fast)
                self._faiss_index = faiss.IndexFlatIP(d)  # Inner product = cosine sim on normalized vectors
                self._faiss_index.add(embeddings)
                print(f"[SemanticMatcher] Built FAISS flat index ({len(self.vocabulary)} words)", flush=True)
                
        except ImportError:
            print("[SemanticMatcher] FAISS not installed, using numpy fallback. Install with: pip install faiss-cpu", flush=True)
            self._faiss_index = None
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get CLIP embedding for an image."""
        import torch
        
        self._ensure_loaded()
        
        if Image is None:
            raise ImportError("PIL is required for image processing")
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = self._clip_preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self._clip_model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def find_similar_words(
        self, 
        image_path: str, 
        top_k: int = 100,
        return_scores: bool = False
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Find top-k most similar words to an image using CLIP.
        Uses FAISS for fast approximate nearest neighbor search when available.
        
        Args:
            image_path: Path to image
            top_k: Number of words to return
            return_scores: If True, return (word, score) tuples
            
        Returns:
            List of words or (word, score) tuples
        """
        self._ensure_loaded()
        
        image_emb = self.get_image_embedding(image_path)
        
        # Use FAISS if available for fast search
        if self._faiss_index is not None:
            import faiss
            # Normalize query for cosine similarity
            query = image_emb.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query)
            
            # Search top-k nearest neighbors
            scores, indices = self._faiss_index.search(query, top_k)
            
            if return_scores:
                return [(self.vocabulary[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
            else:
                return [self.vocabulary[i] for i in indices[0]]
        else:
            # Numpy fallback
            similarities = image_emb @ self._word_embeddings.T
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            if return_scores:
                return [(self.vocabulary[i], float(similarities[i])) for i in top_indices]
            else:
                return [self.vocabulary[i] for i in top_indices]
    
    def describe_image_with_llm(
        self,
        image_path: str,
        top_k: int = 100,
        llm_refiner: Optional['LLMArchetypeRefiner'] = None,
        num_final_words: int = 20,
    ) -> str:
        """
        Get image description by:
        1. Finding top-k similar words via CLIP
        2. Having LLM select and compose the most relevant/interesting ones
        
        Args:
            image_path: Path to image
            top_k: Number of candidate words from CLIP
            llm_refiner: LLM to use for curation (uses Cloudflare if None)
            num_final_words: Target number of words in final description
            
        Returns:
            Curated description string
        """
        # Get candidate words
        word_scores = self.find_similar_words(image_path, top_k=top_k, return_scores=True)
        
        # Format for LLM
        word_list = ", ".join([f"{w} ({s:.2f})" for w, s in word_scores[:50]])  # Top 50 with scores
        
        prompt = f"""You are a semantic curator. Given these words that are visually/conceptually similar to an image (with similarity scores), select and combine the {num_final_words} most evocative and interesting words that would best describe the image's essence.

CANDIDATE WORDS (with similarity scores):
{word_list}

INSTRUCTIONS:
1. Choose words that create a coherent, evocative description
2. Prefer rare, specific, and poetic words over common ones
3. Mix concrete and abstract concepts
4. Return ONLY the selected words as a comma-separated list
5. Do not add any other text or explanation

SELECTED WORDS:"""

        if llm_refiner is None:
            # Use simple Cloudflare LLM call
            from enhanced_miner import LLMArchetypeRefiner
            llm_refiner = LLMArchetypeRefiner(backend="cloudflare")
        
        try:
            # Call the appropriate backend method
            messages = [{"role": "user", "content": prompt}]
            if llm_refiner.backend == "cloudflare":
                response = llm_refiner._call_cloudflare(messages, max_tokens=256)
            else:
                response = llm_refiner._call_local(messages, max_tokens=256)
            # Clean up response
            words = [w.strip() for w in response.split(",")]
            words = [w for w in words if w and len(w) > 2]
            return ", ".join(words[:num_final_words])
        except Exception as e:
            print(f"[SemanticMatcher] LLM curation failed: {e}", flush=True)
            # Fallback: return top words directly
            return ", ".join([w for w, s in word_scores[:num_final_words]])
    
    def batch_describe(
        self,
        image_paths: List[str],
        top_k: int = 100,
        llm_refiner: Optional['LLMArchetypeRefiner'] = None,
        num_final_words: int = 20,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, str]:
        """
        Describe multiple images efficiently.
        
        Returns:
            Dict mapping image_path -> description
        """
        self._ensure_loaded()
        results = {}
        
        for i, path in enumerate(image_paths):
            try:
                desc = self.describe_image_with_llm(
                    path, 
                    top_k=top_k, 
                    llm_refiner=llm_refiner,
                    num_final_words=num_final_words
                )
                results[path] = desc
            except Exception as e:
                print(f"[SemanticMatcher] Error describing {path}: {e}", flush=True)
                results[path] = ""
            
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        
        return results


# ============================================================
# Vision LLM Image Describer
# ============================================================

IMAGE_DESCRIPTION_PROMPT = """Describe this image in rich detail for semantic analysis. Focus on:
1. Main subjects and objects (people, animals, things)
2. Actions, movements, or states
3. Setting, environment, atmosphere
4. Colors, textures, patterns
5. Emotional tone or mood
6. Symbolic or archetypal elements (if any)

Be specific and evocative. Use concrete nouns and vivid adjectives.
Write 2-4 sentences. Do not start with "This image shows" or similar."""


class VisionDescriber:
    """Use Cloudflare's vision LLM to generate rich descriptions of images."""
    
    # Cloudflare vision-capable models
    VISION_MODELS = {
        "llava": "@cf/llava-hf/llava-1.5-7b-hf",
        "llava-next": "@cf/llava-hf/llava-v1.6-mistral-7b-instruct",  # Better quality
    }
    
    def __init__(
        self,
        model: str = "llava",
        cloudflare_account_id: Optional[str] = None,
        cloudflare_api_token: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ):
        self.model = self.VISION_MODELS.get(model, model)
        self.cf_account = cloudflare_account_id
        self.cf_token = cloudflare_api_token
        self.prompt = custom_prompt or IMAGE_DESCRIPTION_PROMPT
        self._cache: Dict[str, str] = {}  # path -> description cache
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        import base64
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read and optionally resize large images
        if Image is not None:
            img = Image.open(path)
            # Resize if too large (vision models have limits)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (handle RGBA, P mode, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save to bytes
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
        else:
            # Fallback: read raw bytes
            image_bytes = path.read_bytes()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def describe_image(self, image_path: str, use_cache: bool = True) -> str:
        """
        Generate a text description for an image using vision LLM.
        
        Args:
            image_path: Path to the image file
            use_cache: Whether to use cached descriptions
            
        Returns:
            Text description of the image
        """
        import requests
        
        # Check cache
        if use_cache and image_path in self._cache:
            return self._cache[image_path]
        
        # Load credentials from env or secrets.toml
        env_account, env_token = _load_cloudflare_credentials()
        account_id = self.cf_account or env_account
        api_token = self.cf_token or env_token
        
        if not account_id or not api_token:
            raise ValueError(
                "Cloudflare credentials not configured. "
                "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables "
                "or add them to .streamlit/secrets.toml"
            )
        
        # Convert image to bytes (resize if needed)
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if Image is not None:
            img = Image.open(path)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
        else:
            image_bytes = path.read_bytes()
        
        # Encode as base64 for the JSON payload
        import base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{self.model}"
        
        # Cloudflare Workers AI LLaVA format
        # image: array of integers 0-255 representing bytes
        # prompt: text prompt
        payload = {
            "prompt": self.prompt,
            "image": list(image_bytes),  # Convert bytes to list of integers
            "max_tokens": 256,
        }
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        
        print(f"[Vision] Sending request, image array length: {len(payload['image'])}, first bytes: {payload['image'][:10]}", flush=True)
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if not result.get("success"):
                errors = result.get("errors", [])
                raise RuntimeError(f"Cloudflare API error: {errors}")
            
            # Output field is 'description' according to API schema
            description = result.get("result", {}).get("description", "").strip()
            print(f"[Vision] Got description: {description[:100] if description else 'empty'}...", flush=True)
            
            # Cache the result
            if description:
                self._cache[image_path] = description
            
            return description
            
        except requests.exceptions.Timeout:
            raise RuntimeError("Vision API timeout. Try again or use a smaller image.")
        except requests.exceptions.HTTPError as e:
            # Get more details from the response
            error_detail = ""
            try:
                error_detail = e.response.text[:500]
            except:
                pass
            raise RuntimeError(f"Vision API request failed: {e}. Details: {error_detail}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Vision API request failed: {e}")
    
    def describe_batch(
        self,
        images: List,  # List of ImageItem or paths
        progress_callback: Optional[Callable[[int, int], None]] = None,
        skip_on_error: bool = True,
        max_workers: int = 5,  # Concurrent API calls
    ) -> Dict[str, str]:
        """
        Describe multiple images using concurrent API calls.
        
        Args:
            images: List of ImageItem objects or image file paths
            progress_callback: Optional callback(current, total) for progress updates
            skip_on_error: If True, continue on errors; if False, raise
            max_workers: Number of concurrent API requests (don't exceed rate limits)
            
        Returns:
            Dict mapping image_path -> description
        """
        import concurrent.futures
        import threading
        
        # Extract paths from ImageItem if needed
        image_paths = []
        for img in images:
            if hasattr(img, 'path'):
                image_paths.append(img.path)
            else:
                image_paths.append(str(img))
        
        results = {}
        completed = [0]  # Use list for mutable counter in closure
        lock = threading.Lock()
        
        def describe_one(path: str) -> tuple:
            """Describe a single image, return (path, description or None)."""
            try:
                desc = self.describe_image(path)
                return (path, desc)
            except Exception as e:
                if not skip_on_error:
                    raise
                return (path, None)
        
        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(describe_one, path): path 
                for path in image_paths
            }
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path, desc = future.result()
                
                with lock:
                    completed[0] += 1
                    if desc:
                        results[path] = desc
                    
                    if progress_callback:
                        progress_callback(completed[0], len(image_paths))
        
        return results
    
    def clear_cache(self):
        """Clear the description cache."""
        self._cache.clear()


# ============================================================
# LLM Archetype Refiner
# ============================================================

ARCHETYPE_SYSTEM_PROMPT = """You are a poet-scholar of symbolic systems, archetypes, and mythological patterns.

Your task: transform raw concept clusters into archetypes with DIVERSE, UNUSUAL, EVOCATIVE descriptors.

ABSOLUTE RULES:
1. NO PROPER NOUNS: Never use names of people, places, brands, religions, cultures
2. ZERO REDUNDANCY: A word may appear in ONLY ONE archetype across the entire output
3. NO SEMANTIC OVERLAP within an archetype: avoid synonyms or closely related words
   BAD: "letters, alphabet, vowels" (all about writing symbols)
   GOOD: "inscription, echo, chronicle" (diverse aspects of written tradition)
4. PREFER UNUSUAL VOCABULARY: avoid common words like "experience", "place", "things", "nature"
   Instead use: "phenomenology", "locus", "artifacts", "wilderness"
5. SEMANTIC BREADTH: each archetype's descriptors should span different domains
   Include: objects, actions, qualities, metaphors, textures, sounds, movements

For each cluster, generate:
- NAME: evocative, 1-3 words
- DESCRIPTORS: 10-12 terms with MAXIMUM semantic diversity (no synonyms, no overlap)
- ESSENCE: one poetic sentence

Output ONLY valid JSON."""

ARCHETYPE_USER_TEMPLATE = """Transform these {n_clusters} concept clusters into archetypes.

SOURCE CONCEPTS (use as inspiration, do NOT copy directly):
{clusters_text}

STRICT REQUIREMENTS:
1. ZERO WORD REPETITION across all archetypes. If "breath" is in Archetype 1, it cannot appear anywhere else.
2. NO SEMANTIC CLUSTERS within descriptors:
   - WRONG: "oral, spoken, voice, speech" (all about speaking)
   - RIGHT: "utterance, resonance, larynx, aria" (diverse aspects)
3. AVOID COMMON/GENERIC WORDS
   Replace with unusual alternatives.
4. MAXIMUM DIVERSITY per archetype - descriptors should feel like they come from different domains:
   Some example of domains, but not limited to:
   - A body part 
   - An action/verb 
   - A texture/quality 
   - A metaphor/image 
   - An unusual noun

QUALITY CHECK before output:
- Count: does any word appear twice? → REJECT and revise
- Similarity: are any two descriptors synonyms? → replace one with something from a different domain

Output valid JSON:
{{
  "archetypes": [
    {{
      "name": "ARCHETYPE_NAME",
      "descriptors": ["term1", "term2", ...],
      "essence": "One poetic sentence"
    }}
  ]
}}

JSON only."""


class LLMArchetypeRefiner:
    """Use LLM to refine and name discovered clusters into proper archetypes."""
    
    def __init__(
        self,
        backend: Literal["cloudflare", "local"] = "cloudflare",
        model: str = "@cf/meta/llama-3.1-8b-instruct",
        cloudflare_account_id: Optional[str] = None,
        cloudflare_api_token: Optional[str] = None,
        local_model_loader: Optional[Callable] = None,
    ):
        self.backend = backend
        self.model = model
        self.cf_account = cloudflare_account_id
        self.cf_token = cloudflare_api_token
        self.local_loader = local_model_loader
    
    def _call_cloudflare(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """Call Cloudflare Workers AI."""
        import requests
        
        # Load credentials from env or secrets.toml
        env_account, env_token = _load_cloudflare_credentials()
        account_id = self.cf_account or env_account
        api_token = self.cf_token or env_token
        
        if not account_id or not api_token:
            raise ValueError(
                "Cloudflare credentials not configured. "
                "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables "
                "or add them to .streamlit/secrets.toml"
            )
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{self.model}"
        
        response = requests.post(
            url,
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.7},
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare API error: {data.get('errors', [])}")
        
        return data.get("result", {}).get("response", "").strip()
    
    def _call_local(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """Call local model (requires loader function)."""
        if self.local_loader is None:
            raise ValueError("Local model loader not provided")
        
        tok, mdl = self.local_loader(self.model)
        # Simplified generation (assumes compatible interface)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        
        import torch
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    def refine_directional(
        self, 
        all_concepts: List[str], 
        filters: List[str],
        semantic_spread: float = 0.5
    ) -> Dict[str, List[str]]:
        
        # 1. Prepare Concept Soup (Only single words now)
        unique_concepts = list(set(all_concepts))
        import random
        random.shuffle(unique_concepts)
        concepts_text = ", ".join(unique_concepts[:400]) 

        filters_text = ", ".join(filters)

        prompt = f"""Synthesize Abstract Archetypes.

    FILTERS (The Containers):
    {filters_text}

    CORPUS INGREDIENTS:
    {concepts_text}

    TASK:
    Create one Archetype for each Filter.
    - NAME: Abstract, 1-2 words. NO "THE".
    - DESCRIPTORS: 8-13 single words PER ARCHETYPE. Each descriptor must be a rare or evocative word.
    - NO word may appear in more than one archetype (all descriptors must be unique across archetypes).

    Output JSON only."""

        messages = [
            {"role": "system", "content": DIRECTIONAL_ARCHETYPE_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            if self.backend == "cloudflare":
                response = self._call_cloudflare(messages, max_tokens=2048)
            else:
                response = self._call_local(messages, max_tokens=2048)
            
            return self._parse_directional_response(response, filters)
        except Exception as e:
            print(f"Directional refinement failed: {e}")
            return {f: ["error"] for f in filters}

    def _parse_directional_response(self, response: str, expected_filters: List[str]) -> Dict[str, List[str]]:
        """
        Parse response and enforce cleaning rules: No 'The', No Multi-word descriptors.
        """
        response = response.strip()
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match: response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match: response = match.group(1)
            
        try:
            data = json.loads(response)
            result = {}
            
            for arch in data.get("archetypes", []):
                raw_name = arch.get("name", "").strip().upper()
                
                # RULE 1: Remove leading "THE "
                if raw_name.startswith("THE "):
                    raw_name = raw_name[4:]
                
                # RULE 2: Enforce Single Word Descriptors
                raw_descriptors = [str(d).lower().strip() for d in arch.get("descriptors", [])]
                clean_descriptors = []
                for d in raw_descriptors:
                    # If it has a space, it's a phrase -> Skip it
                    if " " in d:
                        continue
                    # If it's too short, skip it
                    if len(d) < 3:
                        continue
                    clean_descriptors.append(d)
                
                # Enforce Max 13 descriptors (and implied Min through generation)
                clean_descriptors = clean_descriptors[:13]

                if raw_name and clean_descriptors:
                    result[raw_name] = clean_descriptors
            
            return result
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return {}
   
    def _generate_single_archetype(self, cluster_id: int, concepts: List[str], existing_words: set, semantic_spread: float = 0.5) -> Dict:
        """Generate descriptors for a single cluster, avoiding already-used words.
        
        semantic_spread: 0=focused, 1=divergent
        """
        
        avoid_list = ", ".join(sorted(existing_words)[:50]) if existing_words else "none yet"
        
        # Build spread-dependent instructions
        if semantic_spread < 0.3:
            # FOCUSED: Simple name, coherent descriptors, grounded in source
            spread_instruction = """STYLE: GROUNDED & ACCESSIBLE
NAME FORMAT: 1 word preferred, 2 words maximum
- Good names: "Flame", "Trickster", "Roots", "The Vessel", "Breath"
- Bad names: "The Cartographer of Dreams", "Silken Weaver of Time" (too long)
- Bad names: "Chronostasis", "Terraephyra" (invented/cryptic words)

DESCRIPTORS: Everyday words, tightly connected to source concepts
- All descriptors should CLEARLY relate to the theme found in source concepts
- Stay grounded - every word should feel like it belongs to the same semantic field
- Example for fire concepts: "FLAME" → warmth, burning, light, smoke, spark, heat, glow, ember, torch, blaze"""
        elif semantic_spread < 0.7:
            # BALANCED: Simple name, but semantically rich descriptors
            spread_instruction = """STYLE: ABSTRACTED NAME, RICH DESCRIPTORS
NAME FORMAT: 1 word preferred, 2 words maximum  
- Good names: "Threshold", "Marrow", "The Hollow", "Residue", "Patina"
- Bad names: "The Keeper of Forgotten Whispers" (too long)
- AVOID: Keeper, Walker, Weaver, Maker, Teller, Bearer

DESCRIPTORS: Unusual vocabulary but ALL CONNECTED to source concepts
- Use rare/evocative words: gossamer, patina, sinew, cadence, liminal, chthonic
- Even distant descriptors must be JUSTIFIED by something in the source text
- Create unexpected connections WITHIN the source material's themes
- AVOID generic words: feel, land, air, earth, story, growth, nature
- Example for fire concepts: "CRUCIBLE" → incandescence, transmutation, oxidation, vitreous, phosphor, kiln, temper, slag, anneal, forge-breath"""
        else:
            # DIVERGENT: Cryptic name, surrealist descriptors
            spread_instruction = """STYLE: SURREALIST & MYTHOPOIETIC
NAME FORMAT: 1-2 words, etymologically rich or invented

DESCRIPTORS: Transform basic concepts into SPECIFIC, UNUSUAL vocabulary.
The source concepts are just THEMES - never use them directly as descriptors.
Instead, find rare/technical/poetic words that EVOKE those themes obliquely.

TRANSFORMATION EXAMPLES (source concept → good descriptors). These are just examples - do NOT copy them directly:
- "time, past, future" → chronometry, patina, palimpsest, vestige, harbinger, sediment
- "life, death, living" → pulse, carrion, germination, ossify, quickening, dormancy  
- "earth, ground, soil" → loam, stratum, rhizome, telluric, alluvial, subterranean
- "light, dark, shadow" → penumbra, phosphene, chiaroscuro, crepuscular, umbral
- "water, flow, liquid" → meniscus, turbidity, ablution, rivulet, briny, vitreous

Each descriptor should come from a DIFFERENT domain: body part, texture, sound, movement, material, tool, scientific term, archaic word.

FORBIDDEN SIMPLE WORDS

QUALITY CHECK before output - for EACH descriptor ask:
1. Would a 10-year-old know this word? If YES → replace with rarer word
2. Is it in the FORBIDDEN list above? If YES → replace
3. Does it have fewer than 6 letters? Consider replacing with more specific term

GOOD descriptors: patina, sinew, cartilage, amber, ferric, gossamer, tremor, resonance, membrane, filament, viscera, tincture, striation, pellicle, tessellation"""
        
        prompt = f"""Generate an archetype from these concepts.

SOURCE CONCEPTS (cluster {cluster_id}):
{', '.join(concepts[:40])}

CRITICAL: The source concepts above are just THEMES for inspiration.
Do NOT use common/abstract words as descriptors. Transform them into specific, evocative vocabulary.

WORDS ALREADY USED IN OTHER ARCHETYPES (DO NOT USE ANY OF THESE):
{avoid_list}

{spread_instruction}

RULES:
1. NO word from the "already used" list above
2. NO synonyms or variants (wind/winds, breath/breathing)
3. NO proper nouns (names, places, religions, cultures)
4. Name should be 1-2 words only
5. TRANSFORM basic concepts into unusual/specific vocabulary - never copy them directly
6. SINGLE-WORD descriptors only. Maximum 2 two-word phrases per archetype. 

Output JSON only:
{{
  "name": "ARCHETYPE NAME",
  "descriptors": ["term1", "term2", "term3", "term4", "term5", "term6", "term7", "term8", "term9", "term10"],
  "essence": "One poetic sentence"
}}"""

        messages = [
            {"role": "system", "content": "You are creating symbolic archetypes grounded in source material. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        if self.backend == "cloudflare":
            response = self._call_cloudflare(messages, max_tokens=512)
        else:
            response = self._call_local(messages, max_tokens=512)
        
        return self._parse_single_archetype(response)
    
    def _parse_single_archetype(self, response: str) -> Dict:
        """Parse a single archetype response."""
        response = response.strip()
        
        # Handle markdown
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        try:
            data = json.loads(response)
            return data
        except:
            return {}
    
    def _deduplicate_descriptors(self, archetypes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Remove any duplicate/similar words across archetypes."""
        
        # Build word -> archetype mapping
        word_usage = {}
        for name, descriptors in archetypes.items():
            for desc in descriptors:
                word = desc.lower().strip()
                # Also track stems
                stem = word.rstrip('s').rstrip('ing').rstrip('ed')
                if word not in word_usage:
                    word_usage[word] = []
                word_usage[word].append(name)
                if stem != word and stem not in word_usage:
                    word_usage[stem] = []
                if stem != word:
                    word_usage[stem].append(name)
        
        # Find duplicates
        duplicates = {w: names for w, names in word_usage.items() if len(names) > 1}
        
        if not duplicates:
            return archetypes
        
        # Remove duplicates - keep in first archetype, remove from others
        result = {}
        seen_words = set()
        seen_stems = set()
        
        for name, descriptors in archetypes.items():
            clean_descriptors = []
            for desc in descriptors:
                word = desc.lower().strip()
                stem = word.rstrip('s').rstrip('ing').rstrip('ed')
                
                # Check if word or its stem was already used
                if word in seen_words or stem in seen_stems:
                    continue
                
                clean_descriptors.append(word)
                seen_words.add(word)
                seen_stems.add(stem)
            
            result[name] = clean_descriptors
        
        return result
    
    def _llm_refinement_pass(self, archetypes: Dict[str, List[str]], semantic_spread: float = 0.5) -> Dict[str, List[str]]:
        """
        Second LLM pass: refine all archetypes together to remove redundancy,
        balance descriptors, and ensure distinct archetype identities.
        """
        if len(archetypes) < 2:
            return archetypes
        
        # Build the archetypes summary for the LLM
        archetypes_text = "\n".join([
            f"- {name}: {', '.join(descriptors[:12])}"
            for name, descriptors in archetypes.items()
        ])
        
        # Style guidance based on semantic spread
        if semantic_spread < 0.3:
            style_note = "Names must be 1-2 simple words. Descriptors: common, coherent words."
        elif semantic_spread < 0.7:
            style_note = "Names must be 1-2 words. Descriptors: unusual/evocative but connected."
        else:
            style_note = "Names: 1-2 cryptic/etymological words. Descriptors: surrealist but with invisible thematic threads."
        
        prompt = f"""You are a critical editor refining archetypes. Your job is to ELIMINATE REDUNDANCY ruthlessly.

CURRENT ARCHETYPES:
{archetypes_text}

CRITICAL PROBLEMS TO FIX:

1. NAME LENGTH - names must be 1-2 words ONLY:
   - BAD: "The Cartographer of Dreams", "The Silken Weaver" (too long)
   - GOOD: "Threshold", "Marrow", "The Hollow", "Flame"

2. CONCEPTUAL OVERLAP in names - if multiple archetypes use similar metaphors, RENAME them:
   - BAD: "Cartographer" + "Mapmaker" + "Navigator" (all mapping)
   - BAD: "Weaver" + "Spinner" + "Thread" (all weaving)  
   - BAD: "Storyteller" + "Chronicler" + "Tale" (all narrative)
   - FIX: Give each a COMPLETELY DIFFERENT root metaphor
   
4. DESCRIPTOR DUPLICATES:
   - Remove same/similar words across archetypes
   - Check stems: earth/earthen, speak/spoken, grow/growth

STYLE: {style_note}

CRITICAL: You must output EXACTLY {len(archetypes)} archetypes. Do NOT merge or remove any.
Your job is to RENAME and REFINE, not to reduce the count.

RULES:
- Output exactly {len(archetypes)} archetypes (same count as input)
- Each archetype must have a UNIQUE root metaphor
- NO proper nouns
- Each descriptor in exactly ONE archetype
- 8-12 descriptors per archetype

Output valid JSON:
{{
  "archetypes": [
    {{"name": "ARCHETYPE NAME", "descriptors": ["word1", "word2", ...]}},
    ...
  ]
}}

JSON only."""

        messages = [
            {"role": "system", "content": f"You are refining {len(archetypes)} symbolic archetypes. You must output exactly {len(archetypes)} archetypes. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if self.backend == "cloudflare":
                response = self._call_cloudflare(messages, max_tokens=2048)
            else:
                response = self._call_local(messages, max_tokens=2048)
            
            # Parse response
            parsed = self._parse_refinement_response(response)
            # Stricter check: must have at least 80% of original archetypes
            min_required = max(len(archetypes) - 1, int(len(archetypes) * 0.8))
            if parsed and len(parsed) >= min_required:
                return parsed
            else:
                print(f"Refinement pass returned insufficient archetypes, keeping originals")
                return archetypes
                
        except ValueError as e:
            # Credential/config errors should be fatal
            if "credentials" in str(e).lower() or "not configured" in str(e).lower():
                raise
            print(f"LLM refinement pass failed: {e}, keeping original archetypes")
            return archetypes
        except Exception as e:
            print(f"LLM refinement pass failed: {e}, keeping original archetypes")
            return archetypes
    
    def _parse_refinement_response(self, response: str) -> Dict[str, List[str]]:
        """Parse the refinement LLM response."""
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to fix common issues
            response = re.sub(r',\s*}', '}', response)
            response = re.sub(r',\s*]', ']', response)
            try:
                data = json.loads(response)
            except:
                return {}
        
        # Extract archetypes
        result = {}
        archetypes_list = data.get("archetypes", [])
        
        for arch in archetypes_list:
            name = arch.get("name", "").strip().upper()
            descriptors = arch.get("descriptors", [])
            if name and descriptors:
                result[name] = [str(d).strip().lower() for d in descriptors if d]
        
        return result
    
    def refine_clusters(
        self,
        clusters: Dict[int, List[str]],  # cluster_id -> list of concept strings
        cluster_summaries: Optional[Dict[int, str]] = None,  # optional summaries
        semantic_spread: float = 0.5,  # 0=focused/coherent, 1=diverse/divergent
    ) -> Dict[str, List[str]]:
        """
        Take raw clusters and use LLM to derive proper archetypes.
        
        Pipeline:
        1. Per-cluster LLM calls to generate initial archetypes
        2. Programmatic deduplication
        3. Second LLM pass to refine all archetypes together
        
        semantic_spread: Controls diversity of descriptors within each archetype
          - 0.0: Focused/coherent - descriptors cluster around a tight semantic field
          - 0.5: Balanced - mix of related and divergent terms
          - 1.0: Diverse/divergent - descriptors span wildly different domains
        
        Returns: {archetype_name: [descriptors]}
        """
        archetypes = {}
        all_used_words = set()
        
        # PASS 1: Generate each archetype separately
        for cid, concepts in clusters.items():
            try:
                result = self._generate_single_archetype(cid, concepts, all_used_words, semantic_spread)
                
                if result:
                    name = result.get("name", f"CLUSTER_{cid}").strip().upper()
                    descriptors = [str(d).lower().strip() for d in result.get("descriptors", []) if d]
                    
                    if descriptors:
                        archetypes[name] = descriptors
                        # Track all words used
                        for d in descriptors:
                            all_used_words.add(d)
                            # Also add stems
                            stem = d.rstrip('s').rstrip('ing').rstrip('ed')
                            all_used_words.add(stem)
                    else:
                        raise ValueError("No descriptors found in LLM response")
                else:
                    raise ValueError("Empty result from LLM (JSON parse failed)")
            except ValueError as e:
                # Credential/config errors should be fatal
                if "credentials" in str(e).lower() or "not configured" in str(e).lower():
                    raise
                print(f"Error generating archetype for cluster {cid}: {e}")
                # Fallback for other errors
                if concepts:
                    name = f"{concepts[0]} / {concepts[1]}".upper() if len(concepts) >= 2 else concepts[0].upper()
                    archetypes[name] = concepts[:10]
            except Exception as e:
                print(f"Error generating archetype for cluster {cid}: {e}")
                # Fallback
                if concepts:
                    name = f"{concepts[0]} / {concepts[1]}".upper() if len(concepts) >= 2 else concepts[0].upper()
                    archetypes[name] = concepts[:10]
        
        # Programmatic deduplication
        archetypes = self._deduplicate_descriptors(archetypes)
        
        # PASS 2: LLM refinement of all archetypes together
        if len(archetypes) >= 2:
            archetypes = self._llm_refinement_pass(archetypes, semantic_spread)
        
        return archetypes
    
    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response, handling common formatting issues."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            response = re.sub(r',\s*}', '}', response)  # trailing commas
            response = re.sub(r',\s*]', ']', response)
            try:
                data = json.loads(response)
            except:
                raise ValueError(f"Could not parse LLM response as JSON: {e}\nResponse: {response[:500]}")
        
        # Extract archetypes
        result = {}
        archetypes = data.get("archetypes", [])
        
        for arch in archetypes:
            name = arch.get("name", "").strip().upper()
            descriptors = arch.get("descriptors", [])
            if name and descriptors:
                result[name] = [str(d).strip().lower() for d in descriptors if d]
        
        return result


# ============================================================
# Enhanced Archetype Miner
# ============================================================

class EnhancedArchetypeMiner:
    """
    Full-featured archetype miner with:
    - PDF ingestion + smart chunking
    - Folder image scanning
    - Vision LLM image description
    - Embedding-based clustering
    - LLM-powered archetype refinement
    """
    
    def __init__(
        self,
        text_encoder: Callable[[List[str]], np.ndarray],
        image_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
        llm_refiner: Optional[LLMArchetypeRefiner] = None,
        vision_describer: Optional[VisionDescriber] = None,
    ):
        self.text_enc = text_encoder
        self.img_enc = image_encoder
        self.llm_refiner = llm_refiner
        self.vision_describer = vision_describer
        self.corpus = MinerCorpus()
    
    # ---- Ingestion Methods ----
    
    def add_pdf(
        self,
        pdf_path: str | Path,
        chunking: Literal["paragraph", "sentence", "sliding", "semantic"] = "paragraph",
        **chunk_kwargs
    ) -> int:
        """
        Add a PDF to the corpus.
        
        Returns: number of chunks added
        """
        pages = PDFExtractor.extract_text(pdf_path)
        pdf_name = Path(pdf_path).name
        
        added = 0
        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]
            
            # Apply chunking strategy
            if chunking == "paragraph":
                chunks = TextChunker.chunk_paragraphs(text, **chunk_kwargs)
            elif chunking == "sentence":
                chunks = TextChunker.chunk_sentences(text, **chunk_kwargs)
            elif chunking == "sliding":
                chunks = TextChunker.chunk_sliding_window(text, **chunk_kwargs)
            elif chunking == "semantic":
                chunks = TextChunker.chunk_semantic(
                    text, 
                    embedder=lambda t: self.text_enc.encode(t) if hasattr(self.text_enc, 'encode') else self.text_enc(t),
                    **chunk_kwargs
                )
            else:
                chunks = [text]
            
            for i, chunk_text in enumerate(chunks):
                chunk = TextChunk(
                    id=f"{pdf_name}_p{page_num}_c{i}",
                    text=chunk_text,
                    source=pdf_name,
                    page=page_num,
                    chunk_idx=i,
                    strategy=chunking
                )
                self.corpus.text_chunks.append(chunk)
                added += 1
        
        return added
    
    def add_folder_images(
        self,
        folder: str | Path,
        recursive: bool = True,
        max_files: int = 500,
        describe_images: bool = False,
        description_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Add images from a folder to the corpus.
        
        Args:
            folder: Path to the folder containing images
            recursive: Whether to scan subfolders
            max_files: Maximum number of files to add
            describe_images: If True and vision_describer is set, generate descriptions
            description_callback: Optional callback(current, total) for progress
        
        Returns: number of images added
        """
        images = FolderScanner.scan_images(folder, recursive=recursive, max_files=max_files)
        
        # If vision describer is available and requested, describe images
        if describe_images and self.vision_describer and images:
            def progress_cb(curr, total):
                if description_callback:
                    description_callback(curr, total)
            
            descriptions = self.vision_describer.describe_batch(images, progress_callback=progress_cb)
            
            # Update image items with descriptions
            for img in images:
                if img.path in descriptions:
                    img.description = descriptions[img.path]
        
        self.corpus.images.extend(images)
        return len(images)
    
    def add_text(self, text: str, source: str = "direct", chunk: bool = True) -> int:
        """Add raw text to corpus."""
        if chunk:
            chunks = TextChunker.chunk_paragraphs(text)
        else:
            chunks = [text]
        
        # Use existing corpus size to generate unique IDs
        base_idx = len(self.corpus.text_chunks)
        
        for i, chunk_text in enumerate(chunks):
            tc = TextChunk(
                id=f"{source}_{base_idx + i}",
                text=chunk_text,
                source=source,
                chunk_idx=base_idx + i
            )
            self.corpus.text_chunks.append(tc)
        
        return len(chunks)
    
    # ---- Embedding & Clustering ----
    
    def _embed_corpus(
        self, 
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Embed all corpus items.
        
        Args:
            progress_callback: Optional callback(stage, progress) where progress is 0-1
        
        Returns: (ids, embeddings, types)
        """
        ids = []
        embeddings = []
        types = []
        
        # Text chunks
        if self.corpus.text_chunks:
            if progress_callback:
                progress_callback("Embedding text chunks", 0.0)
            
            texts = [c.text for c in self.corpus.text_chunks]
            text_ids = [c.id for c in self.corpus.text_chunks]
            
            print(f"DEBUG: Embedding {len(texts)} text chunks...")
            
            enc = self.text_enc
            if hasattr(enc, 'encode'):
                text_embs = enc.encode(texts)
            else:
                text_embs = enc(texts)
            
            print(f"DEBUG: Raw embeddings shape: {text_embs.shape}, dtype: {text_embs.dtype}")
            
            # Check for NaN or all-zero embeddings
            nan_count = np.isnan(text_embs).any(axis=1).sum()
            zero_count = (np.abs(text_embs).sum(axis=1) < 1e-8).sum()
            if nan_count > 0 or zero_count > 0:
                print(f"DEBUG: WARNING - {nan_count} NaN embeddings, {zero_count} zero embeddings")
            
            text_embs = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8)
            
            print(f"DEBUG: Normalized embeddings - norms range: [{np.linalg.norm(text_embs, axis=1).min():.3f}, {np.linalg.norm(text_embs, axis=1).max():.3f}]")
            
            ids.extend(text_ids)
            embeddings.append(text_embs)
            types.extend(["text"] * len(text_ids))
            
            if progress_callback:
                progress_callback("Text embedding complete", 0.5)
        
        # Images
        if self.corpus.images and self.img_enc is not None:
            if progress_callback:
                progress_callback("Embedding images", 0.5)
            
            img_paths = [img.path for img in self.corpus.images]
            img_ids = [img.id for img in self.corpus.images]
            
            try:
                if hasattr(self.img_enc, 'encode_images'):
                    img_embs = self.img_enc.encode_images(img_paths)
                else:
                    img_embs = self.img_enc(img_paths)
                
                img_embs = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-8)
                
                ids.extend(img_ids)
                embeddings.append(img_embs)
                types.extend(["image"] * len(img_ids))
                
                if progress_callback:
                    progress_callback("Image embedding complete", 0.8)
            except Exception as e:
                print(f"Warning: Image embedding failed: {e}")
        
        if not embeddings:
            return [], np.array([]), []
        
        if progress_callback:
            progress_callback("Combining embeddings", 0.9)
        
        # Combine (pad to same dimension if needed)
        max_dim = max(e.shape[1] for e in embeddings)
        padded = []
        for emb in embeddings:
            if emb.shape[1] < max_dim:
                pad = np.zeros((emb.shape[0], max_dim - emb.shape[1]))
                emb = np.concatenate([emb, pad], axis=1)
            padded.append(emb)
        
        all_embs = np.vstack(padded)
        all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
        
        return ids, all_embs, types
    
    def _build_similarity_graph(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        k: int = 15,
        min_sim: float = 0.1  # Minimum similarity threshold for edges
    ):
        """Build k-NN similarity graph."""
        if nx is None:
            raise ImportError("networkx required for clustering")
        
        n = len(ids)
        print(f"DEBUG: _build_similarity_graph received {n} ids, embeddings shape={embeddings.shape}")
        
        if n == 0:
            return nx.Graph()
        
        # Compute cosine similarity matrix
        S = embeddings @ embeddings.T
        np.fill_diagonal(S, 0.0)
        
        # Debug: check similarity distribution
        upper_tri = S[np.triu_indices(n, k=1)]
        if len(upper_tri) > 0:
            print(f"DEBUG: Similarity stats - min={upper_tri.min():.3f}, max={upper_tri.max():.3f}, mean={upper_tri.mean():.3f}, median={np.median(upper_tri):.3f}")
        
        # Build k-NN adjacency
        kk = min(k, max(1, n - 1))
        print(f"DEBUG: Using k={kk} for k-NN")
        
        A = np.zeros_like(S)
        idx = np.argpartition(-S, kk, axis=1)[:, :kk]
        rows = np.arange(n)[:, None]
        A[rows, idx] = S[rows, idx]
        A = np.maximum(A, A.T)  # Make symmetric
        
        # Apply minimum similarity threshold
        A[A < min_sim] = 0
        
        edges_before_threshold = (A > 0).sum() // 2
        print(f"DEBUG: {edges_before_threshold} potential edges after k-NN, min_sim={min_sim}")
        
        G = nx.Graph()
        for u in ids:
            G.add_node(u)
        
        print(f"DEBUG: Added {G.number_of_nodes()} nodes to graph")
        
        # Add edges
        nz = np.argwhere(A > 0)
        edge_count = 0
        for i, j in nz:
            if i < j:
                G.add_edge(ids[i], ids[j], weight=float(A[i, j]))
                edge_count += 1
        
        print(f"DEBUG: Graph - {G.number_of_nodes()} nodes, {edge_count} edges added")
        
        return G
    
    def _cluster_graph(
        self,
        G,
        min_size: int = 3,
        resolution: float = 1.0
    ) -> Dict[int, List[str]]:
        """Cluster the similarity graph using Louvain community detection."""
        if G.number_of_edges() == 0:
            print("DEBUG: No edges in graph, cannot cluster")
            return {}
        
        try:
            # Try Louvain (built into networkx)
            communities = nx.community.louvain_communities(
                G, weight='weight', resolution=resolution, seed=42
            )
            
            print(f"DEBUG: Louvain found {len(communities)} communities: {[len(c) for c in communities]}")
            
            clusters = {}
            for cid, members in enumerate(communities):
                if len(members) >= min_size:
                    clusters[cid] = list(members)
            
            print(f"DEBUG: After min_size={min_size} filter: {len(clusters)} clusters remain")
            
            return clusters
            
        except Exception as e:
            print(f"DEBUG: Louvain failed: {e}, trying connected components")
            # Fallback: connected components
            clusters = {}
            for cid, comp in enumerate(nx.connected_components(G)):
                comp = list(comp)
                if len(comp) >= min_size:
                    clusters[cid] = comp
            return clusters
    
    def _extract_cluster_concepts(
        self,
        cluster_ids: List[str],
        top_k: int = 30
    ) -> List[str]:
        """Extract key concepts/terms from a cluster's text content."""
        # Gather all text
        id_to_text = {c.id: c.text for c in self.corpus.text_chunks}
        texts = [id_to_text.get(cid, "") for cid in cluster_ids if cid in id_to_text]
        
        # For image items, use descriptions if available, else filenames
        id_to_img = {img.id: img for img in self.corpus.images}
        img_items = [id_to_img.get(cid) for cid in cluster_ids if cid in id_to_img]
        
        # Add image descriptions as text sources
        for img in img_items:
            if img:
                if hasattr(img, 'description') and img.description:
                    # Use vision LLM description if available
                    texts.append(img.description)
                elif hasattr(img, 'filename'):
                    # Fallback: extract words from filename
                    name = img.filename.rsplit('.', 1)[0] if '.' in img.filename else img.filename
                    name_words = re.findall(r'[a-zA-Z]{3,}', name.lower())
                    if name_words:
                        texts.append(" ".join(name_words))
        
        if not texts:
            return []
        
        # Extract noun phrases / key terms
        all_phrases = []
        for text in texts:
            # Simple extraction: words and 2-grams
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower()) # Min length 4 to skip noise
            all_phrases.extend(words)
            # 2-grams
            for i in range(len(words) - 1):
                all_phrases.append(f"{words[i]} {words[i+1]}")
        
        # Count and filter
        counts = Counter(all_phrases)
        
        # Remove function words and fragments (keep thematic words for LLM context)
        stopwords = {
            # Articles, pronouns, prepositions - pure function words
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'are',
            'was', 'were', 'been', 'being', 'has', 'had', 'not', 'but', 'what',
            'all', 'can', 'her', 'his', 'their', 'they', 'which', 'would', 'there',
            'when', 'who', 'will', 'more', 'some', 'into', 'than', 'them', 'these',
            'then', 'its', 'also', 'only', 'other', 'such', 'could', 'about',
            'our', 'one', 'own', 'those', 'between', 'within', 'through', 'each',
            'how', 'way', 'may', 'must', 'like', 'just', 'over', 'very', 'any',
            'both', 'most', 'after', 'before', 'where', 'while', 'does', 'did',
            'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came', 'see',
            'now', 'new', 'even', 'back', 'well', 'much', 'many', 'here', 'still',
            'yet', 'upon', 'thus', 'onto', 'toward', 'towards', 'among', 'across',
            # Pure fillers
            'myself', 'yourself', 'themselves', 'itself', 'ourselves', 
            'something', 'anything', 'everything', 'nothing', 
            'someone', 'anyone', 'everyone',
            # Fragment phrases to avoid
            'with the', 'from the', 'of the', 'in the', 'to the', 'and the',
            'for the', 'on the', 'at the', 'by the', 'as the', 'is the',
            'that the', 'which the', 'from our', 'with our', 'in our',
        }
        
        # Filter and Weight
        filtered = []
        for word, count in counts.most_common(top_k * 10): # Scan deep
            if word in stopwords: continue
            
            # Heuristic for "Hidden Links":
            # We want words that are descriptive but not generic.
            # We prioritize length slightly as a proxy for complexity.
            weight = count * (len(word) ** 0.5) 
            filtered.append((word, weight))
        
        # Sort by calculated weight
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in filtered[:top_k]]
    
    # ---- Main Pipeline ----
    def run_directional(
        self,
        filters: List[str],
        k_neighbors: int = 15,
        min_cluster_size: int = 5,
        resolution: float = 1.0,
        top_concepts_per_cluster: int = 30,
        debug: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, List[str]]:
        """
        Run the mining pipeline but force the results into specific Filters.
        """
        if len(self.corpus) == 0:
            return {}

        # 1. Standard Embedding & Graph building (Reusing logic to get concepts)
        if progress_callback: progress_callback("Embedding corpus", 0.1)
        ids, embeddings, types = self._embed_corpus()
        
        if len(ids) == 0: return {}

        if progress_callback: progress_callback("Analyzing structure", 0.3)
        G = self._build_similarity_graph(ids, embeddings, k=k_neighbors)
        
        if progress_callback: progress_callback("Extracting themes", 0.5)
        # We still cluster to ensure we get a diverse range of concepts from the whole graph
        clusters = self._cluster_graph(G, min_size=min_cluster_size, resolution=resolution)
        
        # 2. Aggregating Concepts
        # Instead of treating clusters separately, we gather a "Soup" of concepts
        all_concepts_soup = []
        for cid, members in clusters.items():
            # Get top concepts for this cluster
            c_concepts = self._extract_cluster_concepts(members, top_k=top_concepts_per_cluster)
            all_concepts_soup.extend(c_concepts)
        
        # If clustering failed to produce enough, just grab raw concepts from random samples
        if len(all_concepts_soup) < 20:
             all_concepts_soup = self._extract_cluster_concepts([c.id for c in self.corpus.text_chunks], top_k=200)

        # 3. Directional LLM Call
        if progress_callback: progress_callback(f"Mapping to {len(filters)} Filters", 0.7)
        
        if self.llm_refiner:
            archetypes = self.llm_refiner.refine_directional(
                all_concepts=all_concepts_soup,
                filters=filters
            )
            return archetypes, G, clusters, ids # Return G/Clusters for visualization
        else:
            # Fallback if no LLM
            return {f: ["llm", "required", "for", "mode"] for f in filters}, G, clusters, ids
    
    def run(
        self,
        k_neighbors: int = 15,
        min_cluster_size: int = 5,
        resolution: float = 1.0,
        use_llm: bool = True,
        top_concepts_per_cluster: int = 30,
        debug: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, List[str]]:
        """
        Run the full mining pipeline.
        
        Args:
            k_neighbors: Number of neighbors for similarity graph
            min_cluster_size: Minimum items per cluster
            resolution: Louvain resolution parameter
            use_llm: Whether to use LLM for archetype refinement
            top_concepts_per_cluster: Max concepts to extract per cluster
            debug: Print debug info to stdout
            progress_callback: Optional callback(stage_name, progress_0_to_1)
        
        Returns: {archetype_name: [descriptors]}
        """
        if len(self.corpus) == 0:
            if debug:
                print("DEBUG: Empty corpus")
            return {}
        
        # 1. Embed corpus
        if progress_callback:
            progress_callback("Embedding corpus", 0.0)
        
        ids, embeddings, types = self._embed_corpus(progress_callback=progress_callback)
        
        if debug:
            print(f"DEBUG: Embedded {len(ids)} items, shape={embeddings.shape if len(ids) > 0 else 'N/A'}")
        
        if len(ids) == 0:
            return {}
        
        # 2. Build similarity graph
        if progress_callback:
            progress_callback("Building similarity graph", 0.3)
        
        G = self._build_similarity_graph(ids, embeddings, k=k_neighbors)
        
        if debug:
            print(f"DEBUG: Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 3. Cluster
        if progress_callback:
            progress_callback("Clustering", 0.5)
        
        clusters = self._cluster_graph(G, min_size=min_cluster_size, resolution=resolution)
        
        if debug:
            print(f"DEBUG: Found {len(clusters)} clusters: {[(cid, len(m)) for cid, m in clusters.items()]}")
        
        if not clusters:
            return {}
        
        # 4. Extract concepts from each cluster
        if progress_callback:
            progress_callback("Extracting concepts", 0.7)
        
        cluster_concepts = {}
        for cid, members in clusters.items():
            concepts = self._extract_cluster_concepts(members, top_k=top_concepts_per_cluster)
            # Even if no concepts extracted, keep the cluster with placeholders
            if not concepts:
                concepts = [f"cluster_{cid}_item_{i}" for i in range(min(5, len(members)))]
            cluster_concepts[cid] = concepts
            if debug:
                print(f"DEBUG: Cluster {cid} ({len(members)} members) -> {len(concepts)} concepts: {concepts[:5]}")
        
        # 5. Refine with LLM or use simple naming
        if progress_callback:
            progress_callback("Refining with LLM" if use_llm else "Naming clusters", 0.85)
        
        if use_llm and self.llm_refiner is not None:
            try:
                if debug:
                    print("DEBUG: Calling LLM refiner...")
                archetypes = self.llm_refiner.refine_clusters(cluster_concepts)
                if debug:
                    print(f"DEBUG: LLM returned {len(archetypes)} archetypes")
                if archetypes:
                    return archetypes
            except Exception as e:
                if debug:
                    print(f"DEBUG: LLM refinement failed: {e}")
                # Fall through to simple naming
        
        # Fallback: simple naming from top concepts
        archetypes = {}
        for cid, concepts in cluster_concepts.items():
            if len(concepts) >= 2:
                name = f"{concepts[0]} / {concepts[1]}".upper()
            elif concepts:
                name = concepts[0].upper()
            else:
                name = f"CLUSTER_{cid}"
            
            archetypes[name] = concepts[:15]
        
        if debug:
            print(f"DEBUG: Final archetypes: {list(archetypes.keys())}")
        
        return archetypes
    
    def get_embeddings_for_viz(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Get embeddings for visualization (UMAP/t-SNE in UI)."""
        return self._embed_corpus()
    
    def clear(self):
        """Clear the corpus."""
        self.corpus = MinerCorpus()
