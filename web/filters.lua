-- Pandoc Lua filter for the online edition (Markdown -> HTML).
--
-- Numbered things get an empty marker <span class="xn" data-k="KIND" data-id="ID"></span>;
-- cross-references [@id] / [-@id] become <a class="xref" data-ids=".." data-mode="cref|ref"></a>.
-- build_site.py numbers the markers per chapter and resolves the references across pages.

local BOX_TITLES = {systemslens = "Systems Lens", keyidea = "Key Idea", pitfall = "Pitfall",
                    napkinmath = "Napkin Math", deepdive = "Deeper Dive"}
local NUMBERED_DIVS = {exercise = "Exercise", proposition = "Proposition", definition = "Definition",
                       example = "Example"}

local function esc(s) return (s:gsub("&", "&amp;"):gsub('"', "&quot;"):gsub("<", "&lt;")) end
local function marker(kind, id)
  return pandoc.RawInline("html", '<span class="xn" data-k="' .. kind .. '" data-id="' .. esc(id or "") .. '"></span>')
end

-- pull a leading [title]{.box-title} paragraph out of a div
local function take_title(d)
  local b = d.content[1]
  if b and b.t == "Para" and #b.content == 1 and b.content[1].t == "Span"
     and b.content[1].classes:includes("box-title") then
    table.remove(d.content, 1)
    return b.content[1].content
  end
  return nil
end

function Header(h)
  if h.classes:includes("unnumbered") then return h end
  local kind = ({[1] = "chapter", [2] = "section", [3] = "subsection"})[h.level]
  if kind then
    h.content:insert(1, marker(kind, h.identifier))
  elseif h.level >= 4 and h.identifier:find(":", 1, true) then
    h.content:insert(1, marker("paragraph", h.identifier))   -- a labelled paragraph: refers to its section
  end
  return h
end

function Figure(f)
  local id = f.identifier
  if id == "" then
    pandoc.walk_block(pandoc.Div(f.content), {Image = function(i) if i.identifier ~= "" then id = i.identifier end end})
    f.identifier = id
  end
  if f.caption and f.caption.long and #f.caption.long > 0 then
    local first = f.caption.long[1]
    if first.content and #first.content > 0 then first.content:insert(1, marker("figure", id)) end
  end
  return f
end

function Table(t)
  if t.caption and t.caption.long and #t.caption.long > 0 then
    local first = t.caption.long[1]
    if first.content then first.content:insert(1, marker("table", "")) end
  end
  return t
end

function CodeBlock(cb)
  local cap = cb.attributes["caption"]
  if not cap then return nil end
  cb.attributes["caption"] = nil
  local id = cb.identifier
  cb.identifier = ""
  local inl = {}
  local doc = pandoc.read(cap, "markdown")
  if doc.blocks[1] and doc.blocks[1].content then inl = doc.blocks[1].content end
  -- the caption itself may contain cross-references
  inl = pandoc.Inlines(inl):walk({Cite = Cite})
  table.insert(inl, 1, marker("listing", id))
  local capdiv = pandoc.Div({pandoc.Plain(inl)}, pandoc.Attr("", {"listing-caption"}))
  return pandoc.Div({capdiv, cb}, pandoc.Attr(id, {"listing"}))
end

function Div(d)
  if d.classes:includes("figure") then
    local last = d.content[#d.content]
    local cap = {}
    if last and last.t == "Para" and #last.content == 1 and last.content[1].t == "Span"
       and last.content[1].classes:includes("figure-caption") then
      cap = last.content[1].content
      table.remove(d.content)
    end
    local fig = pandoc.Figure(d.content, {pandoc.Plain(cap)}, pandoc.Attr(d.identifier, {"multi"}))
    return Figure(fig)
  end
  for _, c in ipairs(d.classes) do
    if BOX_TITLES[c] then
      local t = take_title(d)
      local head = {pandoc.Str(BOX_TITLES[c])}
      if t then table.insert(head, pandoc.Str(": ")); for _, x in ipairs(t) do table.insert(head, x) end end
      table.insert(d.content, 1, pandoc.Para({pandoc.Strong(head)}))
      d.classes:insert("box")
      return d
    end
    if NUMBERED_DIVS[c] then
      local t = take_title(d)
      local head = {pandoc.Strong({marker(c, d.identifier)})}
      if t then
        table.insert(head, pandoc.Space()); table.insert(head, pandoc.Str("("))
        table.insert(head, pandoc.Emph(t)); table.insert(head, pandoc.Str(")"))
      end
      table.insert(head, pandoc.Str("."))
      table.insert(d.content, 1, pandoc.Para(head))
      if c == "exercise" then d.classes:insert("box") end
      return d
    end
    if c == "labbox" then
      local t = take_title(d) or {}
      local head = {marker("lab", d.identifier), pandoc.Str(": ")}
      for _, x in ipairs(t) do table.insert(head, x) end
      table.insert(d.content, 1, pandoc.Para({pandoc.Strong(head)}))
      d.classes:insert("box")
      return d
    end
    if c == "table-float" then
      -- give the table marker inside this div the div's identifier
      return pandoc.walk_block(d, {RawInline = function(r)
        if r.format == "html" and r.text:find('data-k="table" data-id=""', 1, true) then
          r.text = r.text:gsub('data%-id=""', 'data-id="' .. esc(d.identifier) .. '"')
          return r
        end
      end})
    end
  end
  return nil
end

function Cite(c)
  local ids, mode = {}, "cref"
  for _, ct in ipairs(c.citations) do
    table.insert(ids, ct.id)
    if ct.mode == "SuppressAuthor" then mode = "ref" end
  end
  return pandoc.RawInline("html", '<a class="xref" data-ids="' .. esc(table.concat(ids, ";")) ..
                          '" data-mode="' .. mode .. '"></a>')
end

-- display maths: \label{x} -> \tag{@@EQ:x@@} plus an anchor the reference can jump to
function Inlines(inls)
  local out = pandoc.Inlines({})
  for _, el in ipairs(inls) do
    if el.t == "Math" and el.mathtype == "DisplayMath" and el.text:find("\\label{", 1, true) then
      for lab in el.text:gmatch("\\label{([^}]*)}") do
        out:insert(marker("equation", lab))
      end
      el.text = el.text:gsub("\\label{([^}]*)}", "\\tag{@@EQ:%1@@}")
    end
    out:insert(el)
  end
  return out
end

function Image(img)
  img.attributes["loading"] = "lazy"
  if img.src:match("^figures/") then img.classes:insert("tikz") end
  return img
end

function Para(p)
  if #p.content == 0 then return {} end
  return nil
end
