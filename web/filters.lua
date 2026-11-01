-- Pandoc Lua filter for the online edition.

local BOXES = {systemslens=true, keyidea=true, pitfall=true, napkinmath=true,
               deepdive=true, labbox=true, exercise=true}

-- Code listings: turn the `caption` attribute (raw LaTeX) into a real caption.
function CodeBlock(cb)
  local cap = cb.attributes["caption"]
  if not cap then return nil end
  cb.attributes["caption"] = nil
  cb.attributes["label"] = nil
  local id = cb.identifier
  cb.identifier = ""
  local doc = pandoc.read(cap, "latex")
  local inl = {}
  if doc.blocks[1] and doc.blocks[1].content then inl = doc.blocks[1].content end
  local capdiv = pandoc.Div({pandoc.Plain(inl)}, pandoc.Attr("", {"listing-caption"}))
  return pandoc.Div({capdiv, cb}, pandoc.Attr(id, {"listing"}))
end

-- Boxes: tag them with a common class so CSS can style them uniformly.
function Div(d)
  for _, c in ipairs(d.classes) do
    if BOXES[c] then
      d.classes:insert("box")
      return d
    end
  end
  return nil
end

-- Images: lazy loading, figure SVGs get a class for dark-mode treatment.
function Image(img)
  img.attributes["loading"] = "lazy"
  if img.src:match("^figs/") then img.classes:insert("tikz") end
  return img
end

-- Drop paragraphs that became empty after macro removal.
function Para(p)
  if #p.content == 0 then return {} end
  return nil
end
